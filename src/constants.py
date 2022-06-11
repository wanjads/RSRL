train_episodes = 100000
test_episodes = 100000
new_package_prob = 0.5
send_prob = 0.2
energy_weight = 3
moving_average_length = 50                                                          # for plots
gamma = 0.7                                                                         # discount factor for the MDP

epsilon_0 = 1                                                                       # for eps-greedy policy
decay = 1                                                                           # decay factor for epsilon

aoi_cap = 10


# utility function
acceptance_lvl = 1                                                                  # see shen et al. 2014 p. 5
alpha_utility = 0.05                                                                # risk factor


# CVaR
mu = 50                                                                             # risk weight in CVaR
alpha_cvar = 0.1                                                                    # risk factor


# mean variance
mv_risk_factor = 0.5


# stone
stone_risk_factor = 0.1


# semi std deviation
ssd_risk_factor = 0.1
