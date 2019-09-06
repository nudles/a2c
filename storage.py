import torch

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

    def cuda(self):
        self.observations = self.observations.cuda()
        self.rewards = self.rewards.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()

    def insert(self, step, current_obs, action, action_log_prob, reward):
        self.observations[step + 1].copy_(current_obs)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.rewards[step].copy_(reward)

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
