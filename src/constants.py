import numpy as np

max_episodes = 1000
new_package_prob = 0.9
send_prob = 0.9
energy_weight = 1
moving_average_length = 50                                                          # for the loss plot
gamma = 0.9                                                                         # discount factor for the MDP
epsilon_0 = 0.5                                                                     # for eps-greedy policy
decay = 0.99                                                                        # decay factor for epsilon

risk = True                                                                         # risk sensitivity
alpha = 0.2                                                                         # factor in risk function
def risk_function(cost): return np.exp(alpha * cost)
