import os.path
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

from common.agent import Agents
from common.replay_buffer import ReplayBuffer
from common.rollout import RolloutWorker


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.args.pg = (args.algorithm.find("reinforce") > -1) or (args.algorithm.find("coma") > -1)  # policy gradient method
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.result_dir = Path(args.result_dir) / args.algorithm / args.map
        self.init_dirs()
        self.tensorboard = SummaryWriter(args.log_dir)
        self.device = args.device
        self.win_rates = []
        self.episode_rewards = []

    def init_dirs(self):
        LOG_DIR = self.args.log_dir
        if os.path.isdir(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        os.mkdir(LOG_DIR)
        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        epoch, train_step, evaluate_step = 0, 0, 0
        while epoch < self.args.epochs:
            episode, n_step, battle_won, episode_reward = self.rolloutWorker.generate_episode()

            if not self.args.pg:
                self.buffer.push(episode)
                batch_size = min(len(self.buffer), self.args.batch_size)
                mini_batch = self.buffer.sample(batch_size)
                loss = self.agents.train(mini_batch, train_step)
            else:
                loss = self.agents.train([episode], train_step, self.rolloutWorker.epsilon)

            epoch += n_step
            train_step += 1

            self.tensorboard.add_scalar("Train/Reward", episode_reward, train_step)
            self.tensorboard.add_scalar("Train/Epsilon", self.rolloutWorker.epsilon, train_step)
            for key in loss.keys():
                self.tensorboard.add_scalar("Train/" + key, loss[key], train_step)

            if epoch // self.args.evaluate_cycle > evaluate_step:
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                print("win_rate: {}, mean_reward: {}".format(win_rate, episode_reward))
                self.tensorboard.add_scalar("Evaluate/WinRate", win_rate, evaluate_step)
                self.tensorboard.add_scalar("Evaluate/Reward", episode_reward, evaluate_step)
                if self.args.plot:
                    self.plt_result()
                evaluate_step += 1

        self.agents.policy.save_model()

    def evaluate(self):
        num_won, total_reward = 0, 0
        for index in range(self.args.evaluate_episodes):
            _, _, battle_won, episode_reward = self.rolloutWorker.generate_episode(evaluate=True)
            num_won += int(battle_won)
            total_reward += episode_reward
        return num_won / self.args.evaluate_episodes, total_reward / self.args.evaluate_episodes

    def plt_result(self):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel("Epoch")
        plt.ylabel("WinRate")

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel("Epoch")
        plt.ylabel("Reward")

        plt.savefig(self.result_dir / "result.png", format="png")
        np.save(self.result_dir / "win_rates", self.win_rates)
        np.save(self.result_dir / "episode_rewards", self.episode_rewards)
        plt.close()
