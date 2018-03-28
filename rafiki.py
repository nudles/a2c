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

    def __init__(self, timer, rate, T, sigma=0.1, mu=0.01, seed=1):
        # timer, is shared with the env
        # rate, is the reference throughput
        # self.cur_req_id = 1
        self.timer = timer
        self.T = T
        self.mu = mu
        # to make 20% time the arriving rate is larger than the reference throughput (rate)
        # sin(0.5pi-0.2pi)*k + b = r; sin(0.5pi)*k + b = (1+sigma)*r
        self.k = sigma * rate * 5
        self.b = (1 + sigma) * rate - self.k

        self.last_timestamp = timer.now()
        random.seed(seed)
        # the func should range from 0 to 1
        # T = 2*pi/w

    def num_of_new_requests(self, delta):
        # 20% time the arriving rate is larger than the reference throughput
        x = self.timer.now()
        w = 2 * np.pi / self.T
        num = math.sin(w * x) * self.k + self.b
        num = int(max(0, num * (1 + random.gauss(0, self.mu))))
        return int(num * delta)

    def reset(self):
        self.last_timestamp = self.timer.now()

    def get(self):
        """
        return new requests per time according to inner time model
        :return: (request_id, timestamp)
        """

        cur_time = self.timer.now()
        # num_float = integrate.quad(self.func, self.last_timestamp, cur_time)[0]
        # to make the requests unifromly located in [last_timestamp, now)
        num = self.num_of_new_requests(cur_time - self.last_timestamp)
        new_req = [(random.randint(0, 4999),  # hard code the request id range, here
                    random.uniform(self.last_timestamp, cur_time)) for i in range(num)]
        self.last_timestamp = cur_time
        # sort the requests based on inqueue time
        return sorted(new_req, key=lambda tup: tup[1])


class Discrete:
    # only contains the value of batchsz

    def __init__(self, num_output):
        self.n = num_output

# to label the logger of different env
env_id = 0


class Env:

    def __init__(self, requests_gen, timer, batchsz, tau, latency, perf, alpha=0.5, obs_size=50):
        self.requests_gen = requests_gen
        self.timer = timer
        self.batchsz = batchsz  # a list of candidate batchsz
        self.tau = tau  # time limit of each request
        self.alpha = alpha  # coefficient of the overdue requests in the reward function
        self.obs_size = obs_size  # valid queue length for queue feature into the RL model

        self.latency = latency  # a matrix with one row per model, one column per batchsz
        self.perf = perf  # a list performance/accuracy for (ensemble) model

        self.num_models = self.latency.shape[0]
        self.num_batchsz = self.latency.shape[1]
        nbits = int(math.log2(self.num_batchsz))
        assert (1 << nbits) == self.num_batchsz, 'num batchsz must be 2^x'
        assert (1 << self.num_models) - \
            1 == self.perf.shape[0], 'num of models not math perf file'
        self.state_size = 1

        assert len(batchsz) == self.num_batchsz, \
            'batchsz %d not match latency shape' % len(batchsz)
        # 2^(self.num_models) includes all ensemble combinations. we manually
        # exclude the case where no model is selected in the action
        # the action for model selection and for batchsz selection is merged
        # with the first num_models bits for model selection and the last
        # log2(num_batchsz) for batch selection.
        self.action_space = Discrete(
            ((1 << self.num_models) - 1) * self.num_batchsz)

        # the obs space includes the tau, latency for all models and all
        # batchsz, waiting time of each model to finish existing requests and
        # the queue states (queuing time)
        self.observation_space = np.zeros(
            (self.obs_size + self.latency.size + self.num_models + 1, ))

        # self.reset()
        global env_id
        self.logger = logging.getLogger('Rafiki.env-%d' % env_id)
        env_id += 1

    def model_idx_to_model_action(self, model_idx):
        # convert model selection into model action part
        # model_idx is a binary array, pos k = 1 indicating k-th model is selected
        # - 1 is to exclude [0,0,0...0]
        return model_idx.dot(1 << np.arange(model_idx.size)[::-1]) - 1

    def model_action_to_model_idx(self, action):
        # extract model selection from the model action part
        bstr = bin(action + 1)
        pad = [False] * (self.num_models - (len(bstr) - 2)
                         )  # pad with 0 on the left
        model_idx = np.array(pad + [bool(int(x)) for x in bstr[2:]])
        return model_idx

    def accuracy(self, requests, model_idx):
        return self.perf[self.model_idx_to_model_action(model_idx)]

    def parse_action(self, action):
        # parse full action from RL output into model_idx array and batchsz index
        # first num_models bits for model selection; the rest bits for batchsz
        # index; action value for models selection starting from 1 (0 means
        # no model is selected)
        batchsz_idx = action & (self.num_batchsz - 1)
        nbits = int(math.log2(self.num_batchsz))
        model_idx = self.model_action_to_model_idx(action >> nbits)
        return model_idx > 0, batchsz_idx

    def create_action(self, model_idx, batchsz_idx):
        # reverse op of parse_action
        nbits = int(math.log2(self.num_batchsz))
        action = self.model_idx_to_model_action(model_idx) << nbits
        action += batchsz_idx
        return action

    def step(self, action, sync=False):
        """
          :return: obs s1 and cost c1
        """
        model_idx, batchsz_idx = self.parse_action(action)

        # inc the processing time of the selected models
        self.waiting_time[model_idx] += self.latency[model_idx, batchsz_idx]

        batchsz = self.batchsz[batchsz_idx]
        num_overdue = 0
        cur_time = self.timer.now()
        # the latency of this batch of requests depends on the slowest model
        max_waiting_time = np.max(self.waiting_time[model_idx])

        num = min(batchsz, len(self.requests))
        for _, inqueue_time in self.requests[:num]:
            latency = cur_time - inqueue_time + max_waiting_time
            if latency > self.tau:
                num_overdue += 1
        acc = self.accuracy(self.requests[:num], model_idx)

        reward = acc * num - self.alpha * acc * num_overdue

        # printing
        delta = self.timer.now() - self.last_log_time
        self.logr += reward
        self.logt += num
        self.logo += num_overdue
        self.loga += acc * num
        if delta >= 1:
            self.logger.info('time %5.1f, reward %5.1f, acc %5.3f, overdue %5.1f, throughput %5.1f, arr rate %5.1f, queue size %d, batchsz %d' %
                             (self.timer.now(), self.logr / delta,
                              self.loga / self.logt, self.logo / delta,
                              self.logt / delta,
                              self.requests_gen.num_of_new_requests(1),
                              len(self.requests), batchsz))
            self.last_log_time = self.timer.now()
            self.logr, self.logt, self.logo, self.loga = 0, 0, 0, 0

        # update timer to proceed with the next RL iter
        if sync:
            delta = self.waiting_time
            self.timer.tick(np.max(delta))
        else:
            delta = np.min(self.waiting_time)
            self.timer.tick(delta)

        # delta time has passed
        self.waiting_time -= delta
        # delete the dispatched requests from the queue
        self.requests = self.requests[num:]
        # update env queue status with new requests
        self.update_obs()
        # obs, reward, done, _
        return self.obs, reward, False, \
            {'acc': acc, 'overdue': num_overdue, 'batchsz': num,
             'num_models': sum(model_idx), 'time': cur_time}

    def update_obs(self):
        new_req = self.requests_gen.get()
        total_size = len(new_req) + len(self.requests)
        assert total_size < 100 * self.obs_size, \
            'too many requests %d' % total_size
        self.requests.extend(new_req)
        size = min(self.obs_size, total_size)

        # preare obserations for RL algorithm
        self.obs = np.zeros(
            (self.obs_size + self.latency.size + self.num_models + 1,))
        self.obs[0] = self.tau
        self.obs[1:1 + self.latency.size] = self.latency.reshape((-1,))
        offset = 1 + self.latency.size + self.num_models
        self.obs[1 + self.latency.size: offset] = self.waiting_time
        self.obs[offset: offset + size] = self.timer.now() - \
            np.array([r[1] for r in self.requests[:size]])

    def reset(self):
        # must be called after init env
        self.timer.reset()
        self.requests_gen.reset()
        self.requests = []
        self.waiting_time = np.zeros((self.num_models, ))
        self.last_log_time = self.timer.now()
        self.logr, self.logt, self.logo, self.loga = 0, 0, 0, 0
        self.timer.tick(self.tau / 5)
        self.update_obs()
        return self.obs


class Envs(object):
    # a list of envs

    def __init__(self, num_processes, num_models, policy, obs_size, cycle=200):
        self.num_processes = num_processes
        batchsz = range(16, 65, 16)[:num_models]
        latency = np.loadtxt('latency.txt', delimiter=',')[:num_models]
        perf = np.loadtxt('accuracy.txt', delimiter=',')[:num_models]
        max_rate = sum([batchsz[-1] / l[-1] for l in latency])
        # min when all models are running the same data
        min_rate = min([batchsz[0] / l[0] for l in latency])
        logger.info('max process rate: %f' % max_rate)
        logger.info('min process rate: %f' % min_rate)

        tau = np.max(latency) * 2
        # num_of_iters = obs_size / batchsz[-1]
        T = tau * cycle
        logger.info('sin cycle %f' % T)
        # (pi/2 - asin(0.9)) / pi = 1 / 7 is the peak fraction
        self.envs = []
        for i in range(self.num_processes):
            timer = Timer()
            if policy == 'sync':
                requests_gen = RequestGenerator(timer, 5000, min_rate, T)
                env = Env(requests_gen, timer, batchsz,
                          tau, latency, perf, obs_size=obs_size)
            else:
                requests_gen = RequestGenerator(timer, 5000, max_rate, T)
                env = Env(requests_gen, timer, batchsz,
                          tau, latency, perf, obs_size=obs_size)
            self.envs.append(env)
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
    # greedy algorithm
    # choose biggest batchsz and smallest waiting queue
    action = None
    while action is None:
        tick = False
        for k, bs in zip(range(env.num_batchsz)[::-1], env.batchsz[::-1]):
            if bs <= len(env.requests):
                '''
                if bs < len(env.requests) and (k + 1) < env.num_batchsz:
                    action = env.create_action(model_idx, k + 1)
                else:
                '''
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
    while env.timer.now() < stop_time:
        _, r, _, info = step(env, np.ones((env.num_models), dtype=bool), True)


def async_run(env, stop_time):
    # always use all models
    while env.timer.now() < stop_time:
        # choose biggest batchsz and smallest waiting queue
        model_action = np.zeros((env.num_models), dtype=bool)
        model_action[np.argmin(env.waiting_time)] = True
        # print(np.argmin(env.waiting_time))
        _, r, _, info = step(env, model_action, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Request serving policy optimization.')
    parser.add_argument(
        '--policy', choices=['async', 'sync'], default='sync', help='policy')
    parser.add_argument('--obs_size', type=int, default=50,
                        help='observation vector size')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_models', type=int, default=3)
    parser.add_argument('--cycle_len', type=int, default=500)

    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

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
    latency = np.loadtxt('latency.txt', delimiter=',')[:args.num_models]
    perf = np.loadtxt('accuracy.txt', delimiter=',')[:args.num_models]
    max_rate = sum([batchsz[-1] / l[-1] for l in latency])
    # min when all models are running the same data
    min_rate = min([batchsz[0] / l[0] for l in latency])
    logger.info('max process rate: %f' % max_rate)
    logger.info('min process rate: %f' % min_rate)

    tau = np.max(latency) * 2
    # num_of_iters = args.obs_size / batchsz[-1]
    T = tau * args.cycle_len
    logger.info('sin cycle %f' % T)
    # (pi/2 - asin(0.9)) / pi = 1 / 7 is the peak fraction
    timer = Timer()
    if args.policy == 'sync':
        # always select all models
        requests_gen = RequestGenerator(timer, 5000, min_rate, T)
        env = Env(requests_gen, timer, batchsz,
                  tau, latency, perf, obs_size=args.obs_size)
        env.reset()
        sync_run(env, args.epoch * T)
    else:
        # select the free model
        requests_gen = RequestGenerator(timer, 5000, max_rate, T)
        env = Env(requests_gen, timer, batchsz,
                  tau, latency, perf, obs_size=args.obs_size)
        env.reset()
        async_run(env, args.epoch * T)
