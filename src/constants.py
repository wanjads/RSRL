train_episodes = 10000
test_episodes = 10000
new_package_prob = 0.5
send_prob = 0.9
energy_weight = 3
moving_average_length = 50                                                          # for plots
gamma = 0.7                                                                         # discount factor for the MDP

epsilon_0 = 0.9                                                                     # for eps-greedy policy
decay = 0.999                                                                       # decay factor for epsilon

aoi_cap = 100

risky_aoi = 10

basic_monte_carlo_simulation_no = 100
basic_monte_carlo_simulation_length = 5


reinforce_rollout_length = 300                                                        # batches = eps/rrl
no_train_trajectories = 100000

risk_monte_carlo_rollout_length = 10
risk_monte_carlo_trajectories = 10000000
