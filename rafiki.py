import numpy as np
# from scipy import integrate
import random
import math
import argparse
import logging
import pathlib
import datetime

# create logger
logger = logging.getLogger('Rafiki')
logger.setLevel(logging.INFO)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


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

    def __init__(self, timer, val_size, rate, T, sigma=0.1, mu=0.01):
        # self.cur_req_id = 1
        self.timer = timer
        self.val_size = val_size
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.T = T
        self.mu = mu
        # sin(0.5pi-0.2pi)*k + b = r; sin(0.5pi)*k + b = (1+sigma)*r
        self.k = sigma * rate * 5
        self.b = (1 + sigma) * rate - self.k

        self.last_timestamp = timer.now()
        # the func should range from 0 to 1
        # T = 2*pi/w

    def num_of_new_requests(self, delta):
        x = self.timer.now()
        w = 2 * np.pi / self.T
        num = math.sin(w * x) * self.k + self.b
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
        num = self.num_of_new_requests(cur_time - self.last_timestamp)
        new_req = [(random.randint(0, self.val_size - 1),
                    random.uniform(self.last_timestamp, cur_time)) for i in range(num)]
        self.last_timestamp = cur_time
        return sorted(new_req, key=lambda tup: tup[1])


class Discrete:
    # only contains the value of batchsz

    def __init__(self, num_output):
        self.n = num_output


class Env:

    def __init__(self, requests_gen, timer, batchsz, tau, alpha=1, obs_size=500):
        self.requests_gen = requests_gen
        self.timer = timer
        self.batchsz = batchsz  # a list of candidate batchsz
        self.tau = tau
        self.alpha = alpha
        self.obs_size = obs_size

        self.latency = np.loadtxt('latency.txt', delimiter=',')
        self.perf = np.loadtxt('accuracy.txt', delimiter=',')

        self.num_models = self.latency.shape[0]
        self.num_batchsz = self.latency.shape[1]
        nbits = int(math.log2(self.num_batchsz))
        assert (1 << nbits) == self.num_batchsz, 'num batchsz must be 2^x'
        assert (1 << self.num_models) - \
            1 == self.perf.shape[0], 'num of models not math perf file'
        self.state_size = 1
        self.waiting_time = np.zeros((self.num_models, ))

        assert len(
            batchsz) == self.num_batchsz, 'batchsz %d not match latency shape' % len(batchsz)
        # the following two items is be compatible with OpenAI gym setting.
        self.action_space = Discrete(
            ((1 << self.num_models) - 1) * self.num_batchsz)
        self.observation_space = np.zeros(
            (self.obs_size + self.latency.size + self.num_models + 1, ))
        # requests generation model, we use it to generate requests.
        self.requests = []
        self.reset()

    def model_idx_to_model_action(self, model_idx):
        return model_idx.dot(1 << np.arange(model_idx.size)[::-1]) - 1

    def model_action_to_model_idx(self, action):
        bstr = bin(action + 1)
        pad = [False] * (self.num_models - (len(bstr) - 2))
        model_idx = np.array(pad + [bool(int(x)) for x in bstr[2:]])
        return model_idx

    def accuracy(self, requests, model_idx):
        return self.perf[self.model_idx_to_model_action(model_idx)]

    def parse_action(self, action):
        # first num_models bits for model selection; the rest bits for batchsz
        # index; action value for models selection starting from 1 (0 means
        # no model is selected)
        batchsz_idx = action & (self.num_batchsz - 1)
        nbits = int(math.log2(self.num_batchsz))
        model_idx = self.model_action_to_model_idx(action >> nbits)
        return model_idx > 0, batchsz_idx

    def create_action(self, model_idx, batchsz_idx):
        nbits = int(math.log2(self.num_batchsz))
        action = self.model_idx_to_model_action(model_idx) << nbits
        action += batchsz_idx
        return action

    def step(self, action, sync=False):
        """
          :return: obs s1 and cost c1
        """
        model_idx, batchsz_idx = self.parse_action(action)
        self.waiting_time[model_idx] += self.latency[model_idx, batchsz_idx]
        batchsz = self.batchsz[batchsz_idx]
        num_overdue = 0
        cur_time = self.timer.now()
        max_waiting_time = np.max(self.waiting_time[model_idx])
        num = min(batchsz, len(self.requests))
        for _, inqueue_time in self.requests[:num]:
            latency = cur_time - inqueue_time + max_waiting_time
            if latency > self.tau:
                num_overdue += 1
        acc = self.accuracy(self.requests[:num], model_idx)
        self.requests = self.requests[num:]
        reward = acc * num - self.alpha * acc * num_overdue
        if sync:
            delta = self.waiting_time
            self.timer.tick(np.max(delta))
        else:
            delta = np.min(self.waiting_time)
            self.timer.tick(delta)
        self.waiting_time -= delta
        self.update_obs()
        # obs, reward, done, _
        return self.obs, reward, False, {'acc': acc, 'overdue': num_overdue, 'batchsz': num,
                                         'num_models': sum(model_idx), 'time': cur_time}

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

    def reset(self):
        self.timer.reset()
        self.timer.tick(self.tau / 5)
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


def step(env, model_idx, sync):
     # choose biggest batchsz and smallest waiting queue
    action = None
    while action is None:
        tick = False
        for k, bs in zip(range(env.num_batchsz)[::-1], env.batchsz[::-1]):
            if bs <= len(env.requests):
                action = env.create_action(model_idx, k)
                break
            elif len(env.requests) == 0 or \
                    np.max(env.latency[model_idx, k]) + env.timer.now() - env.requests[0][1] + 0.1 < env.tau:
                env.timer.tick(0.1)
                env.update_obs()
                tick = True
                break
        if not tick and action is None:
            env.timer.tick(0.1)
            env.update_obs()

    return env.step(action, sync)


def sync_run(evn, stop_time):
    last_timestamp = env.timer.now()
    reward, throughput, overdue, acc = 0, 0, 0, 0
    while env.timer.now() < stop_time:
        _, r, _, info = step(env, np.ones((env.num_models), dtype=bool), True)
        reward += r
        throughput += info['batchsz']
        overdue += info['overdue']
        acc += info['acc'] * info['batchsz']
        delta = info['time'] - last_timestamp
        if delta >= 1:
            logger.info('time %f, reward %f, acc %f, overdue %f, throughput %f, arr rate %f, queue size %d' %
                        (info['time'], reward / delta, acc / throughput, overdue / delta,
                         throughput / delta, env.requests_gen.num_of_new_requests(1), len(env.requests)))
            last_timestamp = info['time']
            reward = 0
            throughput = 0
            overdue = 0
            acc = 0


def async_run(env, stop_time):
    # always use all models
    last_timestamp = env.timer.now()
    reward, throughput, overdue, acc = 0, 0, 0, 0
    while env.timer.now() < stop_time:
        # choose biggest batchsz and smallest waiting queue
        model_action = np.zeros((env.num_models), dtype=bool)
        model_action[np.argmin(env.waiting_time)] = True
        # print(np.argmin(env.waiting_time))
        _, r, _, info = step(env, model_action, False)
        reward += r
        throughput += info['batchsz']
        overdue += info['overdue']
        acc += info['acc'] * info['batchsz']
        delta = info['time'] - last_timestamp
        if delta >= 1:
            logger.info('time %f, reward %f, acc %f, overdue %f, throughput %f, arr rate %f, queue size %d' %
                        (info['time'], reward / delta, acc / throughput, overdue / delta,
                         throughput / delta, env.requests_gen.num_of_new_requests(1), len(env.requests)))
            last_timestamp = info['time']
            reward = 0
            throughput = 0
            overdue = 0
            acc = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Request serving policy optimization.')
    parser.add_argument(
        '--policy', choices=['rl', 'async', 'sync'], default='rl', help='policy')
    parser.add_argument('--obs_size', type=int, default=500,
                        help='observation vector size')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='variation of the min/max rate')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if not args.debug:
        pathlib.Path('log').mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler('log/server-%s' %
                                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info(args)

    batchsz = range(16, 65, 16)
    latency = np.loadtxt('latency.txt', delimiter=',')
    max_rate = sum([batchsz[-1] / l[-1] for l in latency])
    # min when all models are running the same data
    min_rate = min([batchsz[0] / l[0] for l in latency])
    logger.info('max process rate: %f' % max_rate)
    logger.info('min process rate: %f' % min_rate)

    tau = np.max(latency) * 2
    num_of_iters = args.obs_size / batchsz[-1]
    T = num_of_iters * tau * 10
    logger.info('sin cycle %f' % T)
    # (pi/2 - asin(0.9)) / pi = 1 / 7 is the peak fraction
    timer = Timer()
    if args.policy == 'sync':
        requests_gen = RequestGenerator(timer, 5000, min_rate, T)
        env = Env(requests_gen, timer, batchsz,
                  tau=tau, obs_size=args.obs_size)
        sync_run(env, args.epoch * T)
    else:
        requests_gen = RequestGenerator(timer, 5000, max_rate, T)
        env = Env(requests_gen, timer, batchsz,
                  tau=tau, obs_size=args.obs_size)
        async_run(env, args.epoch * T)
