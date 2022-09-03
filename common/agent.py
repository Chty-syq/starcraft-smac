import numpy as np
import torch

from policy.iql import IQL
from policy.qmix import QMix
from policy.vdn import VDN


class Agents:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.device = args.device
        if args.algorithm == 'iql':
            self.policy = IQL(args)
        elif args.algorithm == 'vdn':
            self.policy = VDN(args)
        elif args.algorithm == 'qmix':
            self.policy = QMix(args)
        else:
            raise Exception("No such algorithm")

    def choose_action(self, agent_id, obs, avail_actions, last_action, epsilon):
        inputs = obs.copy()
        if self.args.last_action:
            last_action_onehot = np.zeros(self.n_actions)
            last_action_onehot[last_action] = 1
            inputs = np.hstack((inputs, last_action_onehot))
        if self.args.share_network:
            identity = np.zeros(self.n_agents)
            identity[agent_id] = 1
            inputs = np.hstack((inputs, identity))

        h = self.policy.eval_h[:, agent_id, :]
        avail_actions_index = np.nonzero(avail_actions)[0]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, obs_dim)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        q_value, self.policy.eval_h[:, agent_id, :] = self.policy.eval_net(inputs, h)
        q_value[avail_actions == 0.0] = -float('inf')

        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_index)
        else:
            action = torch.argmax(q_value)

        return action

    @staticmethod
    def prep_batch_dict(batch):
        batch_dict = {}
        for episode in batch:
            for key in episode.keys():
                if key not in batch_dict.keys():
                    batch_dict[key] = []
                batch_dict[key].append(episode[key])
        for key in batch_dict.keys():
            batch_dict[key] = np.stack(batch_dict[key], axis=0)
        return batch_dict

    def get_episode_length_max(self, batch):
        episode_length_max = 0
        terminated = batch['t']
        episode_num = terminated.shape[0]
        for i in range(episode_num):
            for j in range(self.args.episode_limit):
                if terminated[i, j] == 1:
                    episode_length_max = max(episode_length_max, j + 1)
                    break

        if episode_length_max == 0:
            episode_length_max = self.args.episode_limit

        return episode_length_max

    def train(self, batch, train_step):
        batch = self.prep_batch_dict(batch)
        episode_length = self.get_episode_length_max(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :episode_length]
        loss = self.policy.learn(batch, train_step, episode_length)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model()
        return loss
