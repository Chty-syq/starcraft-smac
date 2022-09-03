import torch.nn as nn
import torch.nn.functional as F


class DRQN(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.args = args
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, args.n_actions)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        h = h.view(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h)
        q = self.fc2(h)
        return q, h
