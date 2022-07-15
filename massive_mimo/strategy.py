import numpy as np
import constants


class Strategy:

    def __init__(self, strategy_type, risk_factor, m):
        # the strategy type determines how the agent will act, especially how it treats risk
        self.strategy_type = strategy_type
        self.risk_factor = risk_factor
        self.map = m

    # the tabular q-learning update dependent on risk sensitivity
    def update(self, old_state, state, action, episode_no):

        print("a strategy update for strategy type " + self.strategy_type + " is not implemented")

    def action(self, state, epsilon):

        action = np.random.rand(constants.M, constants.K)

        return action
