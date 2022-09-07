import torch.nn as nn
import torch.nn.functional as F


class ComaCritic(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.critic_hidden_dim = args.critic_hidden_dim
        self.fc1 = nn.Linear(input_dim, self.critic_hidden_dim)
        self.fc2 = nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim)
        self.fc3 = nn.Linear(self.critic_hidden_dim, args.n_actions)

    def forward(self, inputs):
        res = F.relu(self.fc1(inputs))
        res = F.relu(self.fc2(res))
        res = self.fc3(res)
        return res
