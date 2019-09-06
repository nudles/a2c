import torch
import torch.nn as nn
from distributions import Categorical

class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        x = self(inputs)
        action,probs = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return action, probs, action_log_probs

    def evaluate_actions(self, inputs, actions):
        x = self(inputs)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return action_log_probs, dist_entropy

class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        else:
            raise NotImplementedError

        self.train()

    def forward(self, inputs):
        x = self.a_fc1(inputs)
        x = torch.tanh(x)

        x = self.a_fc2(x)
        x = torch.tanh(x)

        return x
