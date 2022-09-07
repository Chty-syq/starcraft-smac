from pathlib import Path

import torch
import torch.nn.functional as F
from network.drqn import DRQN


class IQL:
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
        self.target_net = DRQN(self.inputs_shape, args=args).to(self.device)
        if args.load_model:
            self.eval_net.load_state_dict(torch.load(self.model_dir / "drqn_params.pkl"))
        self.eval_params = list(self.eval_net.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_params, lr=args.lr)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.eval_h = None
        self.target_h = None
        self.args = args

    def init_hidden(self, num_episode):
        self.eval_h = torch.zeros((num_episode, self.n_agents, self.args.rnn_hidden_dim)).to(self.device)
        self.target_h = torch.zeros((num_episode, self.n_agents, self.args.rnn_hidden_dim)).to(self.device)

    def prediction(self, inputs, agent_id):
        q_values, self.eval_h[:, agent_id, :] = self.eval_net(inputs, self.eval_h[:, agent_id, :])
        return q_values

    def save_model(self):
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.eval_net.state_dict(), self.model_dir / "drqn_params.pkl")
        print("model saved")

    def get_inputs(self, batch, index):
        o, o_next, u = batch["o"], batch["o_next"], batch["u"]
        n_episode = o.shape[0]

        inputs = torch.from_numpy(o[:, index])  # (n_episode, n_agents, obs_dim)
        inputs_next = torch.from_numpy(o_next[:, index])  # (n_episode, n_agents, obs_dim)

        if self.args.last_action:
            if index == 0:
                last_action_onehot = torch.zeros((n_episode, self.n_agents, self.n_actions))
            else:
                last_action = torch.from_numpy(u[:, index - 1])
                last_action_onehot = F.one_hot(last_action, num_classes=self.n_actions)
            inputs = torch.cat([inputs, last_action_onehot], dim=-1)
            last_action = torch.from_numpy(u[:, index])
            last_action_onehot = F.one_hot(last_action, num_classes=self.n_actions)
            inputs_next = torch.cat([inputs_next, last_action_onehot], dim=-1)

        if self.args.share_network:
            identity = torch.eye(self.n_agents).unsqueeze(0).repeat(n_episode, 1, 1)
            inputs = torch.cat([inputs, identity], dim=-1)
            inputs_next = torch.cat([inputs_next, identity], dim=-1)

        inputs = inputs.view(-1, self.inputs_shape).to(self.device)
        inputs_next = inputs_next.view(-1, self.inputs_shape).to(self.device)
        return inputs, inputs_next

    def learn(self, batch, train_step, episode_length):
        # o: (n_episode, episode_length, n_agents, obs_dim)
        # u: (n_episode, episode_length, n_agents)
        # r: (n_episode, episode_length)
        # t: (n_episode, episode_length)
        # au_next: (n_episode, episode_length, n_agents, n_actions)
        # padding: (n_episode, episode_length)
        o, u, r, t = batch["o"], batch["u"], batch["r"], batch["t"]
        o_next, s_next, au_next, padding = batch["o_next"], batch["s_next"], batch["au_next"], batch["padding"]

        num_episode = o.shape[0]
        self.init_hidden(num_episode)

        q_evals, q_targets = [], []
        for index in range(episode_length):
            inputs, inputs_next = self.get_inputs(batch, index)
            q_eval, self.eval_h = self.eval_net(inputs, self.eval_h)
            q_target, self.target_h = self.target_net(inputs_next, self.target_h)
            q_eval = q_eval.view(num_episode, self.n_agents, -1)
            q_target = q_target.view(num_episode, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_evals = torch.stack(q_evals, dim=1)  # (n_episode, episode_length, n_agents, n_actions)
        q_targets = torch.stack(q_targets, dim=1)

        u = torch.from_numpy(u).view(num_episode, episode_length, self.n_agents, 1).to(self.device)
        r = torch.from_numpy(r).view(num_episode, episode_length, 1).repeat(1, 1, self.n_agents).to(self.device)
        t = torch.from_numpy(t).view(num_episode, episode_length, 1).repeat(1, 1, self.n_agents).to(self.device)
        mask = torch.from_numpy(1 - padding).view(num_episode, episode_length, 1).repeat(1, 1, self.n_agents).to(self.device)
        au_next = torch.from_numpy(au_next).to(self.device)

        q_evals = torch.gather(q_evals, dim=-1, index=u).squeeze(dim=-1)

        q_targets[au_next == 0.0] = -9999999
        q_targets = torch.max(q_targets, dim=-1)[0]

        targets = r + self.args.gamma * q_targets * (1 - t)
        td_error = (q_evals - targets.detach()) * mask

        loss = (td_error**2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.tau == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        return {"Loss": loss.item()}
