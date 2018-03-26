import time
import numpy as np
from scipy import integrate


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

    def __init__(self, timer, val_size, max_rate, min_rate):
        # self.cur_req_id = 1
        self.timer = timer
        self.last_timestamp = timer.now()

        # the func should range from 0 to 1
        # T = 2*pi/w
        w = 2 * np.pi / 20
        self.func = lambda x: math.sin(
            w * x) * (self.max_rate - self.min_rate) + 0.5 * (self.max_rate + self.min_rate) + np.random.normal(scale=0.2)

        self.val_size = val_size
        self.max_rate = max_rate
        self.min_rate = min_rate

    def get(self):
        """
        return new requests per time according to inner time model
        :return: (request_id, timestamp)
        """

        # we should keep the avg generating speed around processing model's maximum processing speed, so as to force
        # the processing is able to process most of the request but can not
        # process all requests.
        num_float = integrate.quad(
            self.func, self.last_timestamp, self.timer.now())[0]
        # to make num between 0 and 1 has 0.5 prob to be 1
        num = int(np.around(num_float))
        self.num_float = num_float

        # to make the requests unifromly located in [last_timestamp, now)
        now = self.timer.now()
        new_req = [(random.randint(0, self.val_size - 1), np.random.randint(self.last_timestamp,
                                                                            now-1) for i in range(num)]
        # update
        self.last_timestamp=self.timer.now()
        return sorted(new_req, key=lambda tup: tup[1])


class Action_Space:
    # only contains the value of batchsz
    shape=(1,)


class Rafiki:

    def __init__(self, batchsz, latency, perf, tau):
        self.queue=deque()
        self.waiting_time=np.zeros((latency.shape[0]))

        # the following two items is be compatible with OpenAI gym setting.
        self.action_space=Action_Space()
        self.observation_space=np.array([0] * self.max_req_num)

        # we use self-defined timer
        self.timer=Timer(0)

        # when using max_batchsz, the processing model retain the maximum processing speed, which will be set as
        # an average speed for genearating model, to force the processing model
        # tend to take maximum batchsz
        self.max_rate=max([batchsz[-1] / l[-1] for l in self.latency])
        self.min_rate=min([batchsz[0] / l[0] for l in self.latency])
        print('max process rate:', self.max_rate)
        print('min process rate:', self.min_rate)

        # requests generation model, we use it to generate requests.
        self.requests_model=RequestModel(
            self.timer, 5000, self.max_rate, self.min_rate)
        self.reset()

    def state(self):
    	return self.state


    def parse_action(self, action):
        

    def step(self, action):
        """
          :return: state s1 and cost c1
        """

        model_index=action[:latency.shape[0]]
        batchsize_index=action[latency.shape[0]:]
        self.waiting_time[
            model_index] += self.latency[model_index, batchsize_index]
        num_overdue=0
        imgs=[]
        for i in range(self.batchsize[batchsize_index]):
            img_id, inqueue_time=self.queue.popleft()
            latency=self.time - inqueue_time + \
                np.max(self.waiting_time[model_index])
            if latency > tau:
                num_overdue += 1
                imgs.append(img_id)
        reward=self.accuracy(imgs, model_index) - self.alpha * num_overdue
        duration=np.min(self.waiting_time)
        self.timer.tick(duration)
        self.waiting_time -= duration
        state=self.update_state(duration)
        # state, reward, done, _
        return state, reward, False, num_overdue

    def update_state(self):
    	self.queue.extend(self.requests_model.get())
        queue_time=self.tick - np.array([r.inqueue_time for r in self.queue])

    def accuracy(self, imgs, model_index):
        comb_id=model_index.dot(1 << np.arange(model_index.size)[::-1])
        return self.perf[comb_id] * len(imgs)

    def reset(self):
    	self.timer.reset()
    	self.timer.tick(100)
        return self.update_state(self)
