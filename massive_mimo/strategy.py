import numpy as np
import constants


class Strategy:

    def __init__(self, strategy_type, risk_factor, m):
        # the strategy type determines how the agent will act, especially how it treats risk
        self.strategy_type = strategy_type
        self.risk_factor = risk_factor
        self.map = m

        # TODO
        # next step: think about a parameterization of the strategy
        # then try to use REINFORCE
        # IMPORTANT: power constraint from the paper, switch reward from max sum to max min

    # the tabular q-learning update dependent on risk sensitivity
    def update(self, old_state, state, action, episode_no):
        pass
        # print("a strategy update for strategy type " + self.strategy_type + " is not implemented")

    def action(self, state, epsilon):

        action = 1 / constants.K * np.ones(constants.K)

        return action
