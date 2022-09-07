from ast import Num
from multiprocessing.pool import IMapUnorderedIterator
from pathlib import Path

import torch
import torch.nn.functional as F
from common import utils
from network.coma_critic import ComaCritic
from network.drqn import DRQN


class COMA:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        self.state_shape = args.state_shape
        self.device = args.device
        self.model_dir = Path(args.model_dir) / args.algorithm / args.map
        self.lambda_ = args.lambda_
        self.gamma = args.gamma
        self.args = args

        self.actor_shape = self.obs_shape
        self.actor_shape += int(args.last_action) * self.n_actions
        self.actor_shape += int(args.share_network) * self.n_agents
        self.critic_shape = self.state_shape + self.obs_shape + 2 * self.n_agents * self.n_actions + self.n_agents

        self.actor = DRQN(self.actor_shape, args).to(self.device)
        self.critic_eval = ComaCritic(self.critic_shape, args).to(self.device)
        self.critic_target = ComaCritic(self.critic_shape, args).to(self.device)

        if args.load_model and self.model_dir.exists():
            self.actor.load_state_dict(torch.load(self.model_dir / "actor_params.pkl"))
            self.critic_eval.load_state_dict(torch.load(self.model_dir / "critic_params.pkl"))
            print("load model success")

        self.critic_target.load_state_dict(self.critic_eval.state_dict())

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic_eval.parameters())
        self.optimizer_actor = torch.optim.RMSprop(self.actor_params, lr=args.lr_actor)
        self.optimizer_critic = torch.optim.RMSprop(self.critic_params, lr=args.lr_critic)

        self.actor_h = None

    def init_hidden(self, num_episode):
        self.actor_h = torch.zeros((num_episode, self.n_agents, self.args.rnn_hidden_dim)).to(self.device)

    def save_model(self):
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), self.model_dir / "actor_params.pkl")
        torch.save(self.critic_eval.state_dict(), self.model_dir / "critic_params.pkl")
        print("model saved")

    def prediction(self, inputs, agent_id):
        probs, self.actor_h[:, agent_id, :] = self.actor(inputs, self.actor_h[:, agent_id, :])
        return probs

    def td_lambda_target(self, q_targets, r, t, mask):
        # q_targets, r, t, mask: (n_episode, episode_len, n_agents)
        n_episode, episode_length = q_targets.shape[:2]
        G = torch.zeros(n_episode, episode_length, self.n_agents, episode_length).to(self.device)
        # G[t, 0] = R[t] + gamma * Q
        # G[t, n] = R[t] + gamma * G[t + 1, n - 1]
        # 倒序递推计算 G[t, n]
        for index in reversed(range(episode_length)):
            G[:, index, :, 0] = (r[:, index] + self.gamma * q_targets[:, index] * (1 - t[:, index])) * mask[:, index]
            for n in range(1, episode_length - index):
                G[:, index, :, n] = (r[:, index] + self.gamma * G[:, index + 1, :, n - 1]) * mask[:, index]

        G_lambda = torch.zeros(n_episode, episode_length, self.n_agents).to(self.device)
        for index in range(episode_length):
            G_lambda[:, index] = pow(self.lambda_, episode_length - index - 1) * G[:, index, :, episode_length - index - 1]
            for n in range(1, episode_length - index):
                G_lambda[:, index] += pow(self.lambda_, n - 1) * (1 - self.lambda_) * G[:, index, :, n]

        return G_lambda

    def get_critic_inputs(self, batch, index):
        inputs, inputs_next = [], []
        n_episode, episode_length = batch["o"].shape[0], batch["o"].shape[1]
        o, s, o_next, s_next = batch["o"][:, index], batch["s"][:, index], batch["o_next"][:, index], batch["s_next"][:, index]

        # state
        s = s.unsqueeze(1).repeat(1, self.n_agents, 1)
        s_next = s_next.unsqueeze(1).repeat(1, self.n_agents, 1)
        inputs.append(s)
        inputs_next.append(s_next)
        # observation
        inputs.append(o)
        inputs_next.append(o_next)
        # last actions for all agents
        u_onehot = F.one_hot(batch["u"][:, index], self.n_actions)
        u_onehot_last = F.one_hot(batch["u"][:, index - 1], self.n_actions) if index > 0 else torch.zeros_like(u_onehot)
        u_onehot_next = F.one_hot(batch["u"][:, index + 1], self.n_actions) if index < episode_length - 1 else torch.zeros_like(u_onehot)
        u_onehot = u_onehot.view(n_episode, 1, -1).repeat(1, self.n_agents, 1)
        u_onehot_last = u_onehot_last.view(n_episode, 1, -1).repeat(1, self.n_agents, 1)
        u_onehot_next = u_onehot_next.view(n_episode, 1, -1).repeat(1, self.n_agents, 1)
        inputs.append(u_onehot_last)
        inputs_next.append(u_onehot)
        # current actions for all agents except self
        action_mask = 1 - torch.eye(self.n_agents)
        action_mask = action_mask.view(1, self.n_agents, self.n_agents, 1).repeat(n_episode, 1, 1, self.n_actions)
        action_mask = action_mask.view(n_episode, self.n_agents, -1).to(self.device)
        inputs.append(u_onehot * action_mask)
        inputs_next.append(u_onehot_next * action_mask)
        # agent_id onehot
        agent_onehot = torch.eye(self.n_agents).unsqueeze(0).repeat(n_episode, 1, 1).to(self.device)
        inputs.append(agent_onehot)
        inputs_next.append(agent_onehot)

        inputs = torch.cat(inputs, dim=-1).view(-1, self.critic_shape)
        inputs_next = torch.cat(inputs_next, dim=-1).view(-1, self.critic_shape)
        return inputs, inputs_next

    def train_critic(self, batch, train_step, episode_length):
        # o: (n_episode, episode_length, n_agents, obs_dim)
        # u: (n_episode, episode_length, n_agents)
        # r: (n_episode, episode_length)
        # t: (n_episode, episode_length)
        # au_next: (n_episode, episode_length, n_agents, n_actions)
        # mask: (n_episode, episode_length)
        o, s, u, au, r, t = batch["o"], batch["s"], batch["u"], batch["au"], batch["r"], batch["t"]
        o_next, s_next, au_next, mask = batch["o_next"], batch["s_next"], batch["au_next"], 1 - batch["padding"]

        n_episode = o.shape[0]
        q_evals, q_targets = [], []
        for index in range(episode_length):
            inputs, inputs_next = self.get_critic_inputs(batch, index)
            q_eval = self.critic_eval(inputs)
            q_target = self.critic_target(inputs_next)
            q_eval = q_eval.view(n_episode, self.n_agents, -1)
            q_target = q_target.view(n_episode, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)

        q_values = q_evals.detach().clone()

        u_next = torch.cat((u[:, 1:], torch.zeros_like(u[:, -1]).unsqueeze(1)), dim=1)
        u_next = u_next.view(n_episode, episode_length, self.n_agents, 1)
        u = u.view(n_episode, episode_length, self.n_agents, 1)
        r = r.view(n_episode, episode_length, 1).repeat(1, 1, self.n_agents)
        t = t.view(n_episode, episode_length, 1).repeat(1, 1, self.n_agents)
        mask = mask.view(n_episode, episode_length, 1).repeat(1, 1, self.n_agents)
        q_evals = torch.gather(q_evals, dim=-1, index=u).squeeze(dim=-1)
        q_targets = torch.gather(q_targets, dim=-1, index=u_next).squeeze(dim=-1)
        # targets = r + self.args.gamma * q_targets * (1 - t)
        targets = self.td_lambda_target(q_targets, r, t, mask)
        td_error = (q_evals - targets.detach()) * mask

        loss = (td_error**2).sum() / mask.sum()
        self.optimizer_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.optimizer_critic.step()

        if train_step > 0 and train_step % self.args.tau == 0:
            self.critic_target.load_state_dict(self.critic_eval.state_dict())

        return q_values, loss.item()

    def get_actor_inputs(self, batch, index):
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

        inputs = inputs.view(-1, self.actor_shape).to(self.device)
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
            inputs = self.get_actor_inputs(batch, index)
            probs, self.actor_h = self.actor(inputs, self.actor_h)
            probs = probs.view(n_episode, self.n_agents, -1)
            probs = F.softmax(probs, dim=-1)
            action_probs.append(probs)
        action_probs = torch.stack(action_probs, dim=1)  # (n_episode, episode_len, n_agents, n_actions)
        action_probs = action_probs * (1 - epsilon) + torch.ones_like(action_probs) * epsilon / n_avail_actions
        action_probs[au == 0] = 0.0
        action_probs = action_probs / torch.sum(action_probs, dim=-1, keepdim=True)
        action_probs[au == 0] = 0.0

        q_values, critic_loss = self.train_critic(batch, train_step, episode_length)  # (n_episode, episode_len, n_agents, n_actions)

        u = u.view(n_episode, episode_length, self.n_agents, 1)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.n_agents)
        q_taken = torch.gather(q_values, dim=-1, index=u).squeeze(dim=-1)
        pi_taken = torch.gather(action_probs, dim=-1, index=u).squeeze(dim=-1)  # (n_episode, episode_len, n_agents)
        pi_taken[mask == 0] = 1.0

        baseline = torch.sum(q_values * action_probs, dim=-1, keepdim=False)
        advantage = q_taken - baseline

        loss = -(advantage.detach() * torch.log(pi_taken) * mask).sum() / mask.sum()
        self.optimizer_actor.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
        self.optimizer_actor.step()
        return {"ActorLoss": loss.item(), "CriticLoss": critic_loss}
