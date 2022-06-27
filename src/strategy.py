import copy
import random
import constants
import numpy as np
import utils
import bisect
import network


class Strategy:

    def __init__(self, strategy_type, risk_factor):
        # the strategy type determines how the agent will act, especially how it treats risk
        self.strategy_type = strategy_type
        self.risk_factor = risk_factor

        # the nn for the strategy
        self.nn = network.NN(3, "mean_squared_error")

        # TODO aufräumen --v

        # cvar needs a sorted list of all past costs
        self.sorted_costs = []

        # mean_variance needs a running avg of all costs, running variance of all costs
        self.mean = 0

        # these strategies need extra information
        if strategy_type == "stone_measure":
            self.target = constants.energy_weight + 1
        if strategy_type == "basic_monte_carlo" or strategy_type == "REINFORCE":
            self.lambda_estimate = 0.5
            self.p_estimate = 0.5
            self.no_of_sends = 0
            self.last_aoi_receiver = 1
            self.last_aoi_sender = 0
        if strategy_type == "REINFORCE":
            self.gradients = []
            self.states = []
            self.costs = []
            self.probs = []

            # the tabular q-learning update dependent on risk sensitivity
    def update(self, old_state, state, action, learning_rate, episode_no, action_probs=None):

        cost = constants.energy_weight * action + state.aoi_receiver
        bisect.insort(self.sorted_costs, cost)
        inp = old_state.as_input()

        if self.strategy_type == "utility_function":  # see shen et al. p.9 eq (13) for tabular version
            utility = utils.utility_function(cost, self.risk_factor)
            self.nn.train_model(inp, action, utility)

        # zhou et al. only use the cost from aoi but in this context using the general cost makes more sense
        elif self.strategy_type == "cvar":  # see zhou et al.
            risk = utils.cvar_risk(self.sorted_costs, self.risk_factor)
            cvar_cost = cost + 1 * risk  # mu seems to be irrelevant
            self.nn.train_model(inp, action, cvar_cost)

        elif self.strategy_type == "mean_variance":  # own idea
            self.mean = utils.running_mean(episode_no, self.mean, cost)
            mv_cost = cost + self.risk_factor * abs(cost - self.mean)
            self.nn.train_model(inp, action, mv_cost)

        elif self.strategy_type == "stone_measure":  # own idea
            stone_cost = cost + self.risk_factor * (cost - self.target) ** 2
            self.nn.train_model(inp, action, stone_cost)

        elif self.strategy_type == "semi_std_deviation":  # own idea
            self.mean = utils.running_mean(episode_no, self.mean, cost)
            if cost < self.mean:
                sd_cost = cost
            else:
                sd_cost = cost + self.risk_factor * (cost - self.mean) ** 2
            self.nn.train_model(inp, action, sd_cost)

        elif self.strategy_type == "risk_neutral" or self.strategy_type == "stochastic":  # standard q-learning
            self.nn.train_model(inp, action, cost)

        elif self.strategy_type == "risk_states":  # own idea
            risky_state_cost = cost
            if state.aoi_receiver >= constants.risky_aoi:
                risky_state_cost = self.risk_factor * cost
            self.nn.train_model(inp, action, risky_state_cost)

        elif self.strategy_type == "basic_monte_carlo":  # own idea
            self.update_estimates(state, episode_no)

        elif self.strategy_type == "REINFORCE":  # standard REINFORCE algorithm, see Sutton and Barto
            self.update_estimates(state, episode_no)
            self.remember(state, action, action_probs, cost)
            if episode_no % constants.reinforce_rollout_length == 0 and episode_no > 0:
                self.update_policy(learning_rate)
        else:
            print("a strategy update for strategy type " + self.strategy_type + " is not implemented")

    def action(self, state, epsilon):
        action_probs = []
        if self.strategy_type == 'always':
            action = 1
        elif self.strategy_type == 'never':
            action = 0
        elif self.strategy_type == 'benchmark':
            action = int(state.aoi_sender == 0)
        elif self.strategy_type == 'benchmark2':
            action = 0
            if constants.send_prob * (state.aoi_receiver - state.aoi_sender - 1) \
                    * (1 + max(0, constants.energy_weight - state.aoi_receiver)
                       + constants.new_package_prob / (1 - constants.new_package_prob) ** 2) \
                    >= constants.energy_weight:
                action = 1
        elif self.strategy_type == 'basic_monte_carlo':
            mean_wait = 0
            mean_send = 0
            for _ in range(constants.basic_monte_carlo_simulation_no):
                mean_wait += self.simulate(constants.basic_monte_carlo_simulation_length, state, "random", 0)
                mean_send += self.simulate(constants.basic_monte_carlo_simulation_length, state, "random", 1)
            mean_wait /= constants.basic_monte_carlo_simulation_no
            mean_send /= constants.basic_monte_carlo_simulation_no
            action = int(mean_send < mean_wait)
        elif self.strategy_type == "cvar":
            action = np.argmin(self.nn.out(state.as_input())[0])
        elif self.strategy_type == "REINFORCE":
            action_probs = self.nn.out(state.as_input())[0]
            action = np.random.choice([0, 1], p=action_probs)
        elif random.random() < epsilon:
            action = random.randint(0, 1)
        elif self.strategy_type == 'stochastic':
            action = 0
            if random.random() < self.nn.out(state.as_input())[0][1]:
                action = 1
        else:
            action = np.argmin(self.nn.out(state.as_input())[0])

        return action, action_probs

    def simulate(self, length, state, simulation_type, action):

        state = copy.deepcopy(state)
        mean = 0
        for simulation_episode in range(length):
            self.simulation_state_update(state, action)
            mean += constants.energy_weight * action + state.aoi_receiver
            if simulation_type == "random":
                action = random.randint(0, 1)

        mean /= length

        return mean

    def simulation_state_update(self, state, action):
        state.last_action = 0
        if action:
            state.last_action = 1
            if random.random() < self.lambda_estimate and state.aoi_sender < constants.aoi_cap:
                state.aoi_receiver = state.aoi_sender

        if state.aoi_receiver < constants.aoi_cap:
            state.aoi_receiver += 1

        # here begins a new iteration

        if state.aoi_sender < constants.aoi_cap:
            state.aoi_sender += 1
        if random.random() < self.p_estimate:
            state.aoi_sender = 0

    def update_estimates(self, state, episode_no):
        self.p_estimate = (int(state.aoi_sender == 0) + episode_no * self.p_estimate) / (episode_no + 1)
        if state.last_action and not self.last_aoi_receiver == self.last_aoi_sender:
            if state.aoi_receiver <= self.last_aoi_receiver:
                self.lambda_estimate = (1 + self.no_of_sends * self.lambda_estimate) / (self.no_of_sends + 1)
            else:
                self.lambda_estimate = self.no_of_sends * self.lambda_estimate / (self.no_of_sends + 1)
            self.no_of_sends += 1
        self.last_aoi_receiver = state.aoi_receiver
        self.last_aoi_sender = state.aoi_sender

    def remember(self, state, action, action_prob, cost):
        self.gradients.append(self.one_hot_encode(action) - action_prob)
        self.states.append(state.as_input())
        self.costs.append(cost)
        self.probs.append(action_prob)

    @staticmethod
    def one_hot_encode(action):
        if action == 0:
            return np.array([1, 0])
        return np.array([0, 1])

    # taken from
    # https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555
    def update_policy(self, learning_rate):
        """Updates the policy network using the NN model.
        This function is used after the MC sampling is done - following
        delta theta = alpha * gradient + log pi"""

        # get X
        states = np.vstack(self.states)

        # get Y
        gradients = np.vstack(self.gradients)
        costs = np.vstack(self.costs)
        gradients *= costs
        gradients = learning_rate * np.vstack([gradients]) + self.probs

        history = self.nn.model.train_on_batch(states, gradients)

        self.states, self.probs, self.gradients, self.costs = [], [], [], []

        return history
