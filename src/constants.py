train_episodes = 100000
test_episodes = 10000
new_package_prob = 0.5
send_prob = 0.5
energy_weight = 1
moving_average_length = 50                                                          # for plots
gamma = 0.9                                                                         # discount factor for the MDP

epsilon_0 = 1                                                                       # for eps-greedy policy
decay = 1                                                                           # decay factor for epsilon

aoi_cap = 10


# utility function
acceptance_lvl = 1                                                                  # see shen et al. 2014 p. 5
alpha_utility = 0.05                                                                # risk factor


# CVaR
mu = 1                                                                              # risk weight in CVaR
alpha_cvar = 0.05                                                                   # risk factor


# mean variance
mv_risk_factor = 1


# stone
stone_risk_factor = 1


# semi std deviation
ssd_risk_factor = 1
