import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape

        self.hyper_w1 = nn.Linear(self.state_shape, self.qmix_hidden_dim * self.n_agents)
        self.hyper_w2 = nn.Linear(self.state_shape, self.qmix_hidden_dim)
        self.hyper_b1 = nn.Linear(self.state_shape, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_shape, self.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.qmix_hidden_dim, 1),
        )

    def forward(self, q, state):
        # q: (n_episode, episode_len, n_agents)
        # state: (n_episode, episode_len, state_dim)
        state = state.view(-1, self.state_shape)
        w1 = torch.abs(self.hyper_w1(state)).view(-1, self.n_agents, self.qmix_hidden_dim)
        w2 = torch.abs(self.hyper_w2(state)).view(-1, self.qmix_hidden_dim, 1)
        b1 = self.hyper_b1(state).view(-1, 1, self.qmix_hidden_dim)
        b2 = self.hyper_b2(state).view(-1, 1, 1)

        n_episode = q.shape[0]
        res = q.view(-1, 1, self.n_agents)
        res = torch.bmm(res, w1) + b1  # (batch, 1, hidden_dim)
        res = F.elu(res)
        res = torch.bmm(res, w2) + b2  # (batch, 1, 1)
        res = res.view(n_episode, -1)
        return res
