import random
import constants
import numpy as np
import utils
import bisect


class Strategy:

    def __init__(self, strategy_type):
        # the strategy type determines how the agent will act, especially how it treats risk
        self.strategy_type = strategy_type

        # tabular saved q values
        self.qvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, 2, 2))

        # cvar needs a sorted list of all past costs
        self.sorted_costs = []

        # mean_variance needs a running avg of all costs, running variance of all costs
        self.mean = 0
        self.var = 0

        if strategy_type == "always":
            self.always_send()
        elif strategy_type == "never":
            self.never_send()
        elif strategy_type == "benchmark":
            self.send_if_new_package()

    # the tabular q-learning update dependent on risk sensitivity
    def update(self, old_state, state, action, learning_rate, episode_no):

        cost = constants.energy_weight * action + state.aoi_receiver
        # in zhou they only  use the cost from aoi but in this context using the general cost makes more sense
        bisect.insort(self.sorted_costs, cost)
        V = np.min(self.qvalues[state.aoi_sender][state.aoi_receiver][state.last_action])
        old_q_value = self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action]

        if self.strategy_type == "utility_function":  # see shen et al. p.9 eq (13)
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] = \
                old_q_value + learning_rate * \
                (utils.utility_function(cost + constants.gamma * V - old_q_value) - constants.acceptance_lvl)

        elif self.strategy_type == "cvar":  # see zhou et al.
            risk = utils.cvar_risk(self.sorted_costs)
            cvar_cost = state.aoi_receiver + constants.mu * risk + constants.energy_weight * action
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (cvar_cost + constants.gamma * V)

        elif self.strategy_type == "mean_variance":  # variance as risk indicator
            old_mean = self.mean
            old_var = self.var
            self.mean = utils.running_mean(episode_no, self.mean, cost)
            if episode_no > 0:
                self.var = utils.running_var(episode_no + 1, old_mean, self.mean, self.var, cost)
            mv_cost = cost + constants.mv_risk_factor * (self.var - old_var)
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (mv_cost + constants.gamma * V)

        elif self.strategy_type == "stone_measure":
            stone_cost = cost + constants.stone_risk_factor*(cost - 1)**2
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (stone_cost + constants.gamma * V)

        elif self.strategy_type == "semi_std_deviation":
            self.mean = utils.running_mean(episode_no, self.mean, cost)
            sd_cost = cost + constants.ssd_risk_factor * (cost - self.mean)**2
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (sd_cost + constants.gamma * V)

        elif self.strategy_type == "risk_neutral":  # standard q-learning
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (cost + constants.gamma * V)

        else:
            print("a strategy update for strategy type " + self.strategy_type + " is not implemented")

    def action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmin(self.qvalues[state.aoi_sender][state.aoi_receiver][state.last_action])
        return action

    def always_send(self):
        for aois in range(constants.aoi_cap + 1):
            for aoir in range(constants.aoi_cap + 1):
                for la in range(2):
                    self.qvalues[aois][aoir][la][1] = -1

    def never_send(self):
        pass  # this happens automatically

    def send_if_new_package(self):
        for aoir in range(constants.aoi_cap + 1):
            for la in range(2):
                self.qvalues[0][aoir][la][1] = -1
