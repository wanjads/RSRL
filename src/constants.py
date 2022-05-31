import numpy as np

train_episodes = 100000
test_episodes = 10000
new_package_prob = 0.5
send_prob = 0.5
energy_weight = 1
moving_average_length = 50                                                          # for plots
gamma = 0.9                                                                         # discount factor for the MDP

epsilon_0 = 1                                                                       # for eps-greedy policy
decay = 1                                                                           # decay factor for epsilon

aoi_cap = 5

alpha = 0.25                                                                        # factor in risk function
def risk_function(cost): return np.exp(alpha * cost)


def learning_rate(episode): return 10000000 / (10000000 + episode)                  # leaning factor for tab. q learn
