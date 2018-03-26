import time
import numpy as np
from scipy import integrate
import random


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

    def num_of_new_requests(self, x, delta)
        w = 2 * np.pi / self.T
        num = (math.sin(w * x)) * (self.max_rate - self.min_rate) / 2 + 0.5 * (self.max_rate + self.min_rate)
        num = int(max(0, num * (1 + random.gauss(0, self.mu))))
        return num * delta

    def get(self):
        """
        return new requests per time according to inner time model
        :return: (request_id, timestamp)
        """

        # we should keep the avg generating speed around processing model's maximum processing speed, so as to force
        # the processing is able to process most of the request but can not
        # process all requests.
        cur_time = self.timer.now()
        # num_float = integrate.quad(self.func, self.last_timestamp, cur_time)[0]
        # to make the requests unifromly located in [last_timestamp, now)
        num = self.num_of_new_requests(cur_time, cur_time - self.last_timestamp)
        new_req = [(random.randint(0, self.val_size - 1), random.uniform(self.last_timestamp, cur_time)) for i in range(num)]
        self.last_timestamp=cur_time
        return sorted(new_req, key=lambda tup: tup[1])


class Action_Space:
    # only contains the value of batchsz
    shape=(1,)


class Rafiki:

    def __init__(self, batchsz, latency, perf, state_size=500):
        self.requests=[]
        self.num_models = latency.shape[0]
        self.state_size = state_size
        self.batchsz = batchsz  # a list of candidate batchsz
        self.waiting_time=np.zeros((num_models, ))

        # the following two items is be compatible with OpenAI gym setting.
        self.action_space=Action_Space()
        self.observation_space=np.array([0] * self.state_size)

        # we use self-defined timer
        self.timer=Timer(0)

        # max when all models are running diff data
        self.max_rate=sum([batchsz[-1] / l[-1] for l in self.latency])
        # min when all models are running the same data
        self.min_rate=min([batchsz[0] / l[0] for l in self.latency])
        print('max process rate:', self.max_rate)
        print('min process rate:', self.min_rate)

        self.tau = np.max(latency) * 2
        num_of_iters = self.state_size / batchsz[-1]
        T = num_of_iters * tau * 10
        # (pi/2 - asin(0.9)) / pi = 1 / 7 is the peak fraction

        # requests generation model, we use it to generate requests.
        self.requests_gen=RequestGenerator(self.timer, 5000, self.max_rate, self.min_rate, T)
        self.reset()

    def state(self):
    	return self.state

    def parse_action(self, action):
        # first num_models bits for model selection; the rest bits for batchsz index
        b = [int(x) for x in bin(action)[2:]]
        model_idx = b[0:len(self.num_models)]
        batchsz_idx = np.array(b[len(num_models):])
        batchsz_idx=batchsz_idx.dot(1 << np.arange(batchsz_idx.size)[::-1])
        return np.where(model_idx>0)[0], batchsz_idx

    def step(self, action):
        """
          :return: state s1 and cost c1
        """
        model_idx, batchsz_idx=self.parse_action(action)
        self.waiting_time[model_idx] += self.latency[model_idx, batchsz_idx]
        batchsz = self.batchsz[batchsz_idx]
        num_overdue=0
        cur_time = self.timer.now()
        max_waiting_time = np.max(self.waiting_time[model_idx])
        for _, inqueue_time in self.requests[:batchsz]:
            latency= cur_time - inqueue_time + max_waiting_time
            if latency > self.tau:
                num_overdue += 1
        acc = self.accuracy(self.requests[:batchsz],  model_idx)
        self.requests= self.requests[batchsz:]
        reward = acc * batchsz - self.alpha * num_overdue
        delta=np.min(self.waiting_time)
        self.timer.tick(delta)
        self.waiting_time -= delta
        self.update_state()
        # state, reward, done, _
        return self.state, reward, False, {'acc': acc, 'overdue': num_overdue}

    def update_state(self):
        new_req = self.requests_gen.get()
        total_size = len(new_req) + len(self.requests)
        assert total_size < 10*self.state_size, 'too many requests %d' % total_size
    	self.requests.extend(new_req)
        size = min(self.state_size, total_size)
        self.state = np.zeros((self.state_size + self.latency.size + 1,))
        self.state[:self.latency.size] = self.latency.reshape((-1,))
        self.state[self.latency.size + 1] = self.tau
        self.state[self.latency.size + 1 :] = self.timer.now() - np.array([r[1] for r in self.requests[:size]])

    def accuracy(self, requests, model_idx):
        comb_id=model_idx.dot(1 << np.arange(model_idx.size)[::-1])
        return self.perf[comb_id]

    def reset(self):
    	self.timer.reset()
    	self.timer.tick(self.tau)
        return self.update_state(self)
