import random

import numpy as np
import torch
from smac.env import StarCraft2Env

from common.arguments import get_args
from runner import Runner

if __name__ == '__main__':
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子

    env = StarCraft2Env(map_name=args.map, difficulty=args.difficulty)
    env_info = env.get_env_info()

    args.n_actions = env_info['n_actions']
    args.n_agents = env_info['n_agents']
    args.state_shape = env_info['state_shape']
    args.obs_shape = env_info['obs_shape']
    args.episode_limit = env_info['episode_limit']
    args.device = torch.device('cuda' if torch.cuda else 'cpu')

    print('using device ' + str(args.device))

    runner = Runner(env, args)

    if not args.eval:
        runner.train()
    else:
        win_rate, _ = runner.evaluate()
        print("win_rate: {}".format(win_rate))

    env.close()
