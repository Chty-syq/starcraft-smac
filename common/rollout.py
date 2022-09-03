import time

import numpy as np
import torch


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.state_shape = args.state_shape
        self.args = args

        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.anneal_steps = args.anneal_steps
        self.epsilon_anneal = (self.epsilon - self.epsilon_min) / self.anneal_steps

    @torch.no_grad()
    def generate_episode(self, evaluate=False):
        self.env.reset()
        self.agents.policy.init_hidden(1)

        n_step = 0
        episode_reward = 0
        terminated = False
        battle_won = False
        o, u, r, s, t, avail_u, padding = [], [], [], [], [], [], []
        last_action = np.zeros(self.n_agents, dtype=int)

        epsilon = 0 if evaluate else self.epsilon

        while n_step < self.episode_limit and not terminated:
            obs = self.env.get_obs()  # (n_agents, obs_dim)
            state = self.env.get_state()  # (state_dim)
            if self.args.render:
                self.env.render()
                time.sleep(0.1)
            if epsilon - self.epsilon_anneal > self.epsilon_min:
                epsilon = epsilon - self.epsilon_anneal

            actions, avail_actions = [], []

            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(agent_id, obs[agent_id], avail_action, last_action[agent_id],
                                                   epsilon)

                last_action[agent_id] = action
                actions.append(np.int(action))
                avail_actions.append(avail_action)

            reward, terminated, info = self.env.step(actions)

            o.append(obs)
            u.append(actions)
            s.append(state)
            r.append(reward)
            t.append(terminated)
            avail_u.append(avail_actions)
            padding.append(0)

            episode_reward += reward
            battle_won = True if terminated and 'battle_won' in info and info['battle_won'] else False
            n_step += 1

        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # padding for rnn
        for i in range(n_step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros(self.n_agents))
            s.append(np.zeros(self.state_shape))
            r.append(0.0)
            t.append(True)
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padding.append(1)

        o = np.array(o, dtype=np.float32)
        u = np.array(u, dtype=np.int)
        s = np.array(s, dtype=np.float32)
        r = np.array(r, dtype=np.float32)
        t = np.array(t, dtype=np.int)
        o_next = np.array(o_next, dtype=np.float32)
        s_next = np.array(s_next, dtype=np.float32)
        avail_u = np.array(avail_u, dtype=np.int)
        avail_u_next = np.array(avail_u_next, dtype=np.int)
        padding = np.array(padding, dtype=np.int)

        episode = dict(o=o, u=u, s=s, r=r, t=t, s_next=s_next, o_next=o_next, au=avail_u, au_next=avail_u_next,
                       padding=padding)

        if not evaluate:
            self.epsilon = epsilon

        return episode, n_step, battle_won, episode_reward
