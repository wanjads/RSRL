import random
import constants
import numpy as np


class Strategy:

    def __init__(self, risk):
        self.risk_sensitivity = risk
        self.constant = False
        self.qvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, 2, 2))

    def update(self, old_state, state, action, learning_rate):

        self.constant = False
        cost = constants.energy_weight * action + state.aoi_receiver
        V = np.min(self.qvalues[state.aoi_sender][state.aoi_receiver][state.last_action])
        old_q_value = self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action]

        if self.risk_sensitivity:  # see shen et al. p.9 eq (13)
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] = \
                old_q_value + learning_rate * \
                (constants.utility_function(cost + constants.gamma * V - old_q_value) - constants.acceptance_lvl)
        else:  # standard q-learning
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] = \
                (1 - learning_rate) * old_q_value + learning_rate * (cost + constants.gamma * V)

    def action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmin(self.qvalues[state.aoi_sender][state.aoi_receiver][state.last_action])
        return action

    def send_always(self):
        self.constant = True
        for aois in range(constants.aoi_cap):
            for aoir in range(constants.aoi_cap):
                for la in range(2):
                    self.qvalues[aois][aoir][la][0] = 0
                    self.qvalues[aois][aoir][la][0] = 1
