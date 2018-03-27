import numpy as np
# from scipy import integrate
import random
import math


class Timer:
    """
    This is a man-hand driven timer. The user should manually tick the timer.
    """

    def __init__(self, start_time=0):
        self.time = start_time

    def tick(self, delta):
        assert delta >= 0
        self.time += delta

        return self.time

    def now(self):
        return self.time

    def reset(self, start_time=0):
        self.time = start_time


class RequestGenerator(object):

    def __init__(self, timer, val_size, max_rate, min_rate, T, mu=0.1):
        # self.cur_req_id = 1
        self.timer = timer
        self.val_size = val_size
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.T = T
        self.mu = mu

        self.last_timestamp = timer.now()
        # the func should range from 0 to 1
        # T = 2*pi/w

    def num_of_new_requests(self, x, delta):
        w = 2 * np.pi / self.T
        num = (math.sin(w * x)) * (self.max_rate - self.min_rate) / \
            2 + 0.5 * (self.max_rate + self.min_rate)
        num = int(max(0, num * (1 + random.gauss(0, self.mu))))
        return int(num * delta)

    def get(self):
        """
        return new requests per time according to inner time model
        :return: (request_id, timestamp)
        """

        cur_time = self.timer.now()
        # num_float = integrate.quad(self.func, self.last_timestamp, cur_time)[0]
        # to make the requests unifromly located in [last_timestamp, now)
        num = self.num_of_new_requests(
            cur_time, cur_time - self.last_timestamp)
        new_req = [(random.randint(0, self.val_size - 1),
                    random.uniform(self.last_timestamp, cur_time)) for i in range(num)]
        self.last_timestamp = cur_time
        return sorted(new_req, key=lambda tup: tup[1])


class Discrete:
    # only contains the value of batchsz

    def __init__(self, num_output):
        self.n = num_output


class Env:

    def __init__(self, batchsz, latency_path, perf_path, alpha=0, obs_size=500):
        self.alpha = alpha
        self.latency = np.loadtxt(latency_path, delimiter=',')
        self.perf = np.loadtxt(perf_path, delimiter=',')
        self.num_models = self.latency.shape[0]
        nbits = int(math.log2(self.latency.shape[1]))
        assert (1 << nbits) == self.latency.shape[1], 'num batchsz must be 2^x'
        assert (1 << self.latency.shape[
                0]) - 1 == self.perf.shape[0], 'num of models not math perf file'
        self.obs_size = obs_size
        self.state_size = 1
        self.batchsz = batchsz  # a list of candidate batchsz
        self.waiting_time = np.zeros((self.num_models, ))

        assert len(batchsz) == self.latency.shape[
            1], 'batchsz %d not match latency shape' % len(batchsz)
        # the following two items is be compatible with OpenAI gym setting.
        self.action_space = Discrete(
            ((1 << self.num_models) - 1) * self.latency.shape[1])
        self.observation_space = np.zeros(
            (self.obs_size + self.latency.size + self.num_models + 1, ))

        # we use self-defined timer
        self.timer = Timer(0)

        # max when all models are running diff data
        self.max_rate = sum([batchsz[-1] / l[-1] for l in self.latency])
        # min when all models are running the same data
        self.min_rate = min([batchsz[0] / l[0] for l in self.latency])
        print('max process rate:', self.max_rate)
        print('min process rate:', self.min_rate)

        self.tau = np.max(self.latency) * 2
        num_of_iters = self.obs_size / batchsz[-1]
        T = num_of_iters * self.tau * 10
        # (pi/2 - asin(0.9)) / pi = 1 / 7 is the peak fraction

        # requests generation model, we use it to generate requests.
        self.requests = []
        self.requests_gen = RequestGenerator(
            self.timer, 5000, self.max_rate, self.min_rate, T)
        self.reset()

    def parse_action(self, action):
        # first num_models bits for model selection; the rest bits for batchsz
        # index
        action = int(action)
        model_part = action // self.latency.shape[1] + 1
        batchsz_idx = action % self.latency.shape[1]
        model_idx = np.array([int(x) for x in bin(model_part)[2:]])
        return np.where(model_idx > 0)[0], batchsz_idx

    def step(self, action):
        """
          :return: obs s1 and cost c1
        """
        model_idx, batchsz_idx = self.parse_action(action)
        self.waiting_time[model_idx] += self.latency[model_idx, batchsz_idx]
        batchsz = self.batchsz[batchsz_idx]
        num_overdue = 0
        cur_time = self.timer.now()
        max_waiting_time = np.max(self.waiting_time[model_idx])
        for _, inqueue_time in self.requests[:batchsz]:
            latency = cur_time - inqueue_time + max_waiting_time
            if latency > self.tau:
                num_overdue += 1
        acc = self.accuracy(self.requests[:batchsz],  model_idx)
        self.requests = self.requests[batchsz:]
        reward = acc * batchsz - self.alpha * num_overdue
        delta = np.min(self.waiting_time)
        self.timer.tick(delta)
        self.waiting_time -= delta
        self.update_obs()
        # obs, reward, done, _
        return self.obs, reward, False, {'acc': acc, 'overdue': num_overdue}

    def update_obs(self):
        new_req = self.requests_gen.get()
        total_size = len(new_req) + len(self.requests)
        assert total_size < 10 * self.obs_size, 'too many requests %d' % total_size
        self.requests.extend(new_req)
        size = min(self.obs_size, total_size)
        self.obs = np.zeros(
            (self.obs_size + self.latency.size + self.num_models + 1,))
        self.obs[0] = self.tau
        self.obs[1:1 + self.latency.size] = self.latency.reshape((-1,))
        offset = 1 + self.latency.size + self.num_models
        self.obs[1 + self.latency.size: offset] = self.waiting_time
        self.obs[offset: offset + size] = self.timer.now() - \
            np.array([r[1] for r in self.requests[:size]])

    def accuracy(self, requests, model_idx):
        comb_id = model_idx.dot(1 << np.arange(model_idx.size)[::-1]) - 1
        return self.perf[comb_id]

    def reset(self):
        self.timer.reset()
        self.timer.tick(self.tau)
        return self.update_obs()


class Envs(object):

    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.envs = [Env(range(16, 65, 16), 'latency.txt',
                         'accuracy.txt', obs_size=500) for i in range(num_processes)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def step(self, actions):
        obs = np.empty((self.num_processes, self.observation_space.shape[0]))
        reward = np.empty((self.num_processes))
        done = np.empty((self.num_processes))
        info = []
        for k, env in enumerate(self.envs):
            o, r, d, s = env.step(actions[k])
            obs[k, :] = o
            reward[k] = r
            done[k] = d
            info.append(s)
        return obs, reward, done, info

    def reset(self):
        ret = np.empty((self.num_processes, self.observation_space.shape[0]))
        for i, env in enumerate(self.envs):
            ret[i, :] = env.reset()
        return ret


def max_strategy(niter=1000):
    env = Env()
    for i in range(niter):
        # choose biggest batchsz and smallest waiting queue
        pass


def min_strategy():
    # always use all models
    pass
