import random
import constants
import numpy as np


class Strategy:

    def __init__(self, risk):
        self.risk_sensitivity = risk
        self.qvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, 2, 2))

    def update(self, old_state, state, action, cost, learning_rate):

        V = np.max(self.qvalues[state.aoi_sender][state.aoi_receiver][state.last_action])
        self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] = \
            (1 - learning_rate) * \
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][old_state.last_action][action] \
            + learning_rate * \
            (cost + constants.gamma * V)

    def action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmin(self.qvalues[state.aoi_sender][state.aoi_receiver][state.last_action])
        return action

    def send_always(self):
        for aois in range(constants.aoi_cap):
            for aoir in range(constants.aoi_cap):
                for la in range(2):
                    self.qvalues[aois][aoir][la][0] = 0
                    self.qvalues[aois][aoir][la][0] = 1