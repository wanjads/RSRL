import utils


class State:

    def __init__(self, sinrs, m):
        self.sinrs = sinrs
        self.map = m

    def update(self, action):
        self.sinrs = utils.sinrs(self.map, action)

    # the initial state
    @staticmethod
    def initial_state(initial_action, m):
        return State(utils.sinrs(m, initial_action), m)
