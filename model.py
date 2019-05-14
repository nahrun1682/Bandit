import numpy as np
import random

class BernoulliArm():

    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0

class random_select():

    def __init__(self, counts, values):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        return random.randint(0, len(self.values) - 1)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value

class EpsilonGreedy():

    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randint(0, len(self.values) - 1)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value

class UCB():

    def __init__(self, counts, values):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        n_arms = len(self.counts)
        if min(self.counts) == 0:
            return np.argmin(self.counts)

        total_counts = sum(self.counts)
        bonus = np.sqrt((np.log(np.array(total_counts))) /
                        2 / np.array(self.counts))
        ucb_values = np.array(self.values) + bonus
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value

class ThompsonSampling():

    def __init__(self, counts_alpha, counts_beta, values):
        self.counts_alpha = counts_alpha
        self.counts_beta = counts_beta
        self.alpha = 1
        self.beta = 1
        self.values = values

    def initialize(self, n_arms):
        self.counts_alpha = np.zeros(n_arms)
        self.counts_beta = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        theta = [(arm,
                  random.betavariate(self.counts_alpha[arm] + self.alpha,
                                     self.counts_beta[arm] + self.beta))
                 for arm in range(len(self.counts_alpha))]
        theta = sorted(theta, key=lambda x: x[1])
        return theta[-1][0]

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.counts_alpha[chosen_arm] += 1
        else:
            self.counts_beta[chosen_arm] += 1
        n = float(self.counts_alpha[chosen_arm]) + self.counts_beta[chosen_arm]
        self.values[chosen_arm] = (n - 1) / n * \
            self.values[chosen_arm] + 1 / n * reward

def test_algorithm(algo, arms, num_sims, horizon):
    chosen_arms = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)
    for sim in range(num_sims):
        algo.initialize(len(arms))
        for t in range(horizon):
            index = sim * horizon + t
            times[index] = t + 1
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm
            reward = arms[chosen_arm].draw()
            if t == 0:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[
                    index - 1] + reward
            algo.update(chosen_arm, reward)
    return [times, chosen_arms, cumulative_rewards]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import (BernoulliArm,
                   random_select,
                   EpsilonGreedy,
                   UCB,
                   ThompsonSampling,
                   test_algorithm)
import random

n_arms = 10
means = [0.054,  0.069,  0.080,  0.097,  0.112,
         0.119,  0.121,  0.144,  0.155,  0.174]

epsilon = 0.2  # パラメータ
sim_num = 500  # シミュレーション回数
time = 10000  # 試行回数

arms = pd.Series(map(lambda x: BernoulliArm(x), means))

algo_1 = random_select([], [])           # random
algo_2 = EpsilonGreedy(epsilon, [], [])  # epsilon-greedy
algo_3 = UCB([], [])                    # UCB
algo_4 = ThompsonSampling([], [], [])   # ThompsonSampling
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
heights = []
random.seed(2017)
for algo in [algo_1, algo_2, algo_3, algo_4]:
    algo.initialize(n_arms)
    result = test_algorithm(algo, arms, sim_num, time)

    df_result = pd.DataFrame({"times": result[0], "chosen_arms": result[1]})
    df_result["best_arms"] = (df_result["chosen_arms"]
                              == np.argmax(means)).astype(int)
    grouped = df_result["best_arms"].groupby(df_result["times"])

    ax1.plot(grouped.mean(), label=algo.__class__.__name__)
    heights.append(result[2][-1])

ax1.set_title("Compare 4model - Best Arm Rate")
ax1.set_xlabel("Time")
ax1.set_ylabel("Best Arm Rate")
ax1.legend(loc="upper left")

plt_label = ["Random", "Epsilon\nGreedy", "UCB", "Tompson \nSampling"]
plt_color = ["deep", "muted", "pastel", "bright"]
ax2.bar(range(1, 5), heights, color=sns.color_palette()[:4], align="center")
ax2.set_xticks(range(1, 5))
ax2.set_xticklabels(plt_label)
ax2.set_label("random_select")
ax2.set_ylabel("Cumulative Rewards")
ax2.set_title("Compare 4model - Cumulative Rewards")
plt.show()
