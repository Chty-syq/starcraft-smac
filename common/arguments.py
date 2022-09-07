import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()

    # game environment option
    parser.add_argument("--map", type=str, default="3m")
    parser.add_argument("--difficulty", type=str, default="7")

    # dirs
    parser.add_argument("--log_dir", type=str, default=Path(__file__).parent.parent / "tensorboard")
    parser.add_argument("--model_dir", type=str, default=Path(__file__).parent.parent / "model")
    parser.add_argument("--result_dir", type=str, default=Path(__file__).parent.parent / "result")

    # training option
    parser.add_argument("--algorithm", type=str, choices=["iql", "vdn", "qmix", "reinforce", "coma"], default="iql")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--share_network", type=bool, default=True)
    parser.add_argument("--last_action", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=2000000)
    parser.add_argument("--epsilon", type=float, default=1)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--anneal_steps", type=int, default=50000)
    parser.add_argument("--anneal_mode", type=str, choices=["step", "episode"], default="step")
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tau", type=int, default=200)
    parser.add_argument("--evaluate_episodes", type=int, default=32)
    parser.add_argument("--evaluate_cycle", type=int, default=5000)
    parser.add_argument("--save_cycle", type=int, default=200)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--rnn_hidden_dim", type=int, default=64)
    parser.add_argument("--grad_norm_clip", type=float, default=10)

    # qmix option
    parser.add_argument("--qmix_hidden_dim", type=int, default=32)

    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=True)

    return parser.parse_args()


def get_reinforce_args(args):
    args.lr = 1e-4
    args.epsilon = 0.5
    args.epsilon_min = 0.02
    args.anneal_steps = 750
    args.anneal_mode = "episode"
    return args


def get_coma_args(args):
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3
    args.critic_hidden_dim = 64
    args.lambda_ = 0.8
    args.epsilon = 0.5
    args.epsilon_min = 0.02
    args.anneal_steps = 750
    args.anneal_mode = "episode"
    return args
