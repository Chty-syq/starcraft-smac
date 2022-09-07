from curses import tparm
from pathlib import Path

import torch
import torch.nn.functional as F
from common import utils
from network.drqn import DRQN


class Reinforce:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        self.state_shape = args.state_shape
        self.device = args.device
        self.model_dir = Path(args.model_dir) / args.algorithm / args.map
        self.inputs_shape = self.obs_shape
        self.inputs_shape += int(args.last_action) * self.n_actions
        self.inputs_shape += int(args.share_network) * self.n_agents

        self.eval_net = DRQN(self.inputs_shape, args=args).to(self.device)
        if args.load_model and self.model_dir.exists():
            self.eval_net.load_state_dict(torch.load(self.model_dir / "drqn_params.pkl"))
            print("load model success")

        self.eval_params = list(self.eval_net.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_params, lr=args.lr)

        self.eval_h = None
        self.args = args

    def init_hidden(self, num_episode):
        self.eval_h = torch.zeros((num_episode, self.n_agents, self.args.rnn_hidden_dim)).to(self.device)

    def prediction(self, inputs, agent_id):
        probs, self.eval_h[:, agent_id, :] = self.eval_net(inputs, self.eval_h[:, agent_id, :])
        return probs

    def save_model(self):
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.eval_net.state_dict(), self.model_dir / "drqn_params.pkl")
        print("model saved")

    def get_inputs(self, batch, index):
        o, u = batch["o"], batch["u"]
        n_episode = o.shape[0]
        inputs = o[:, index]  # (n_episode, n_agents, obs_dim)

        if self.args.last_action:
            if index == 0:
                last_action_onehot = torch.zeros((n_episode, self.n_agents, self.n_actions))
            else:
                last_action = u[:, index - 1]
                last_action_onehot = F.one_hot(last_action, num_classes=self.n_actions)
            last_action_onehot = last_action_onehot.to(self.device)
            inputs = torch.cat([inputs, last_action_onehot], dim=-1)

        if self.args.share_network:
            identity = torch.eye(self.n_agents).unsqueeze(0).repeat(n_episode, 1, 1).to(self.device)
            inputs = torch.cat([inputs, identity], dim=-1)

        inputs = inputs.view(-1, self.inputs_shape).to(self.device)
        return inputs

    def learn(self, batch, train_step, episode_length, epsilon):
        batch = utils.batch_np_to_tensor(batch, self.device)
        # o: (n_episode, episode_length, n_agents, obs_dim)
        # u: (n_episode, episode_length, n_agents)
        # r: (n_episode, episode_length)
        # t: (n_episode, episode_length)
        # au_next: (n_episode, episode_length, n_agents, n_actions)
        # mask: (n_episode, episode_length)
        o, s, u, au, r, t = batch["o"], batch["s"], batch["u"], batch["au"], batch["r"], batch["t"]
        o_next, s_next, au_next, mask = batch["o_next"], batch["s_next"], batch["au_next"], 1 - batch["padding"]

        n_episode = o.shape[0]
        n_avail_actions = torch.sum(au, dim=-1, keepdim=True).repeat(1, 1, 1, au.shape[-1])
        self.init_hidden(n_episode)

        action_probs = []
        for index in range(episode_length):
            inputs = self.get_inputs(batch, index)
            probs, self.eval_h = self.eval_net(inputs, self.eval_h)
            probs = probs.view(n_episode, self.n_agents, -1)
            probs = F.softmax(probs, dim=-1)
            action_probs.append(probs)
        action_probs = torch.stack(action_probs, dim=1)  # (n_episode, episode_len, n_agents, n_actions)
        action_probs = action_probs * (1 - epsilon) + torch.ones_like(action_probs) * epsilon / n_avail_actions
        action_probs[au == 0] = 0.0
        action_probs = action_probs / torch.sum(action_probs, dim=-1, keepdim=True)
        action_probs[au == 0] = 0.0

        Gt = torch.zeros((n_episode, episode_length)).to(self.device)
        Gt[:, -1] = r[:, -1] * mask[:, -1]
        for index in reversed(range(episode_length - 1)):
            Gt[:, index] = (r[:, index] + self.args.gamma * Gt[:, index + 1] * (1 - t[:, index])) * mask[:, index]
        Gt = Gt.unsqueeze(-1).repeat(1, 1, self.n_agents)

        u = u.view(n_episode, episode_length, self.n_agents, 1)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.n_agents)
        pi_taken = torch.gather(action_probs, dim=-1, index=u).squeeze(dim=-1)  # (n_episode, episode_len, n_agents)
        pi_taken[mask == 0] = 1.0

        loss = -(Gt * torch.log(pi_taken) * mask).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.eval_params, self.args.grad_norm_clip)
        self.optimizer.step()
        return {"Loss": loss.item()}
