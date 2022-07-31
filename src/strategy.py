import random
import constants
import numpy as np
import utils
import bisect


class Strategy:

    def __init__(self, strategy_type, risk_factor):
        # the strategy type determines how the agent will act, especially how it treats risk
        self.strategy_type = strategy_type
        self.risk_factor = risk_factor

        # tabular saved q values
        self.qvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, 2))

        # cvar needs a sorted list of all past costs
        self.sorted_costs = []

        # mean_variance needs a running avg of all costs, running variance of all costs
        self.mean = 0

        if strategy_type == "always":
            self.always_send()
        elif strategy_type == "never":
            self.never_send()
        elif strategy_type == "benchmark":
            self.send_if_new_package()
        elif strategy_type == "stone_measure":
            self.target = constants.energy_weight + 1
        elif strategy_type == "value_iteration":
            self.value_iteration()

    # the tabular q-learning update dependent on risk sensitivity
    def update(self, old_state, state, action, learning_rate, episode_no):

        cost = constants.energy_weight * action + state.aoi_receiver
        bisect.insort(self.sorted_costs, cost)
        V = np.min(self.qvalues[state.aoi_sender][state.aoi_receiver])
        old_q_value = self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][action]

        if self.strategy_type == "utility_function":  # see shen et al. p.9 eq (13)
            # acceptance_lvl = 1, see shen et al. 2014 p. 5
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][action] = \
                old_q_value + learning_rate * \
                (utils.utility_function(cost + constants.gamma * V - old_q_value, self.risk_factor) - 1)

        # zhou et al. only use the cost from aoi but in this context using the general cost makes more sense
        elif self.strategy_type == "cvar":  # see zhou et al.
            risk = utils.cvar_risk(self.sorted_costs, self.risk_factor)
            cvar_cost = cost + 1 * risk  # mu seems to be irrelevant
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (cvar_cost + constants.gamma * V)

        elif self.strategy_type == "mean_variance":  # own idea
            self.mean = utils.running_mean(episode_no, self.mean, cost)
            mv_cost = cost + self.risk_factor * abs(cost - self.mean)
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (mv_cost + constants.gamma * V)

        elif self.strategy_type == "stone_measure":  # own idea
            stone_cost = cost + self.risk_factor*(cost - self.target)**2
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (stone_cost + constants.gamma * V)

        elif self.strategy_type == "semi_std_deviation":  # own idea
            self.mean = utils.running_mean(episode_no, self.mean, cost)
            if cost < self.mean:
                sd_cost = cost
            else:
                sd_cost = cost + self.risk_factor * (cost - self.mean)**2
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (sd_cost + constants.gamma * V)

        elif self.strategy_type == "risk_neutral":  # standard q-learning
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (cost + constants.gamma * V)

        else:
            print("a strategy update for strategy type " + self.strategy_type + " is not implemented")

    def action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmin(self.qvalues[state.aoi_sender][state.aoi_receiver])
        return action

    def always_send(self):
        for aois in range(constants.aoi_cap + 1):
            for aoir in range(constants.aoi_cap + 1):
                self.qvalues[aois][aoir][1] = -1

    def never_send(self):
        pass  # this happens automatically

    def send_if_new_package(self):
        for aoir in range(constants.aoi_cap + 1):
            self.qvalues[0][aoir][1] = -1

    def value_iteration(self):

        vvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1))
        qvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, 2))

        eps = 1
        for iteration in range(eps):
            for aois in range(constants.aoi_cap + 1):
                print(aois + 1000 * iteration)
                for aoir in range(constants.aoi_cap + 1):
                    for action in range(2):
                        summe = 0
                        p_summe = 0
                        for aois2 in range(constants.aoi_cap + 1):
                            for aoir2 in range(constants.aoi_cap + 1):
                                if not action:
                                    p = constants.new_package_prob * \
                                        (aois2 == 0 and (aoir2 == aoir + 1 or aoir2 == constants.aoi_cap == aoir))\
                                        + (1 - constants.new_package_prob) * \
                                        ((aois2 == aois + 1 or aois2 == constants.aoi_cap == aois) and
                                         (aoir2 == aoir + 1 or aoir2 == constants.aoi_cap == aoir))
                                    r = aoir2
                                else:
                                    p = constants.new_package_prob * constants.send_prob * \
                                        (aois2 == 0 and (aoir2 == aois + 1 or aoir2 == constants.aoi_cap == aois))\
                                        + constants.new_package_prob * (1 - constants.send_prob) * \
                                        (aois2 == 0 and (aoir2 == aoir + 1 or aoir2 == constants.aoi_cap == aoir))\
                                        + (1 - constants.new_package_prob) * constants.send_prob * \
                                        ((aois2 == aois + 1 or aois2 == constants.aoi_cap == aois) and
                                         (aoir2 == aois + 1 or aoir2 == constants.aoi_cap == aois))\
                                        + (1 - constants.new_package_prob) * (1 - constants.send_prob) * \
                                        ((aois2 == aois + 1 or aois2 == constants.aoi_cap == aois) and
                                         (aoir2 == aoir + 1 or aoir2 == constants.aoi_cap == aoir))
                                    r = aoir2 + constants.energy_weight
                                summe += p * (r + constants.gamma * vvalues[aois2][aoir2])
                                p_summe += p
                        qvalues[aois][aoir][action] = summe
                    vvalues[aois][aoir] = np.min(qvalues[aois][aoir])

        for aois in range(constants.aoi_cap + 1):
            for aoir in range(constants.aoi_cap + 1):
                self.qvalues[aois][aoir][0] = qvalues[aois][aoir][0]
                self.qvalues[aois][aoir][1] = qvalues[aois][aoir][1]