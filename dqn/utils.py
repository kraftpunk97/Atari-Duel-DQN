import numpy as np
import matplotlib.pyplot as plt
import re


def plot_log_rewards(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    rewards = [re.search(r"Reward: (\d*.\d*)", line) for line in lines]
    rewards = np.array([float(match.group(1)) for match in rewards if match is not None],
                       dtype=np.float32)
    X = np.arange(start=25, stop=(len(rewards)+1)*25, step=25)
    plt.plot(X, rewards, color='blue')
    plt.xlabel("Number of episodes")
    plt.ylabel("Mean reward of last 20 episodes")
    plt.savefig("plot.png")


if __name__ == '__main__':
    plot_log_rewards("rl_collections/dqn/logs/Duel-DQN Breakout2.log")
