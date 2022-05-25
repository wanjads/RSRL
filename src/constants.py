import numpy as np

train_episodes = 2000
test_episodes = 10000
new_package_prob = 0.5
send_prob = 0.5
energy_weight = 1
moving_average_length = 50                                                          # for plots
gamma = 0.9                                                                         # discount factor for the MDP
epsilon_0 = 0.6                                                                     # for eps-greedy policy
decay = 0.999                                                                       # decay factor for epsilon

alpha = 0.25                                                                        # factor in risk function
def risk_function(cost): return np.exp(alpha * cost)
