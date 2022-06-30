import copy
import random
import constants
import numpy as np
import utils
import bisect
import network
from state import State


class Strategy:

    def __init__(self, strategy_type, risk_factor):
        # the strategy type determines how the agent will act, especially how it treats risk
        self.strategy_type = strategy_type
        self.risk_factor = risk_factor

        # the nn for the strategy
        self.nn = network.NN(3, "mean_squared_error")

        # these strategies need extra information
        if strategy_type == "mean_variance" or strategy_type == "semi_std_deviation":
            self.mean = 0
        if strategy_type == "cvar":
            self.sorted_costs = []
        if strategy_type == "stone_measure":
            self.target = constants.energy_weight + 1
        if strategy_type == "basic_monte_carlo":
            self.lambda_estimate = 0.5
            self.p_estimate = 0.5
            self.no_of_sends = 0
            self.last_aoi_receiver = 1
            self.last_aoi_sender = 0
        if strategy_type == "REINFORCE":
            self.action_prob = 0.5

            # the tabular q-learning update dependent on risk sensitivity
    def update(self, old_state, state, action, learning_rate, episode_no):

        cost = constants.energy_weight * action + state.aoi_receiver
        inp = old_state.as_input()

        if self.strategy_type == "utility_function":  # see shen et al. p.9 eq (13) for tabular version
            utility = utils.utility_function(cost, self.risk_factor)
            self.nn.train_model(inp, action, utility)

        # zhou et al. only use the cost from aoi but in this context using the general cost makes more sense
        elif self.strategy_type == "cvar":  # see zhou et al.
            bisect.insort(self.sorted_costs, cost)
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

        else:
            print("a strategy update for strategy type " + self.strategy_type + " is not implemented")

    def update_reinforce(self, states, actions, costs, learning_rate):

        returns = self.returns_from_costs(costs)

        derivatives = self.derivatives_from_state_actions(states, actions)

        for episode_no in range(len(states)):
            self.action_prob = \
                max(0, min(1, self.action_prob + learning_rate * derivatives[episode_no] * returns[episode_no]))
            print(self.action_prob)

    def action(self, state, epsilon):
        if self.strategy_type == 'always':
            action = 1
        elif self.strategy_type == 'never':
            action = 0
        elif self.strategy_type == 'random':
            action = random.randint(0, 1)
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
                data = self.simulate(constants.basic_monte_carlo_simulation_length, state, "random", 0)
                mean_wait += self.cost_mean_from_simulation_data(data)
                data = self.simulate(constants.basic_monte_carlo_simulation_length, state, "random", 1)
                mean_send += self.cost_mean_from_simulation_data(data)
            mean_wait /= constants.basic_monte_carlo_simulation_no
            mean_send /= constants.basic_monte_carlo_simulation_no
            action = int(mean_send < mean_wait)
        elif self.strategy_type == "cvar":
            action = np.argmin(self.nn.out(state.as_input())[0])
        elif self.strategy_type == "REINFORCE":
            action_probs = [(1 - self.action_prob), self.action_prob]
            action = np.random.choice([0, 1], p=action_probs)
        elif random.random() < epsilon:
            action = random.randint(0, 1)
        elif self.strategy_type == 'stochastic':
            action = 0
            if random.random() < self.nn.out(state.as_input())[0][1]:
                action = 1
        else:
            action = np.argmin(self.nn.out(state.as_input())[0])

        return action

    def simulate(self, length, state, simulation_type, action):

        state = copy.deepcopy(state)
        data = []
        for simulation_episode in range(length):
            self.simulation_state_update(state, action)
            cost = constants.energy_weight * action + state.aoi_receiver
            if simulation_type == "random":
                action = random.randint(0, 1)
            elif simulation_type == "reinforce":
                action = self.action(state, 0)
            data += [[state.aoi_sender, state.aoi_receiver, state.last_action, action, cost]]

        return data

    @staticmethod
    def cost_mean_from_simulation_data(data):
        mean = 0
        for d in data:
            mean += d[4]
        mean /= len(data)
        return mean

    @staticmethod
    def returns_from_costs(costs):

        returns = -(costs - np.mean(costs)) / np.std(costs)

        # reverse the array
        returns = returns[::-1]

        for i in range(len(returns)):
            if i > 0:
                returns[i] += returns[i - 1]

        # reverse the array
        returns = returns[::-1]

        return returns

    def derivatives_from_state_actions(self, states, actions):

        derivatives = np.zeros(constants.reinforce_rollout_length)

        for episode_no in range(len(states)):
            if actions[episode_no]:
                derivatives[episode_no] = 1 / self.action_prob
            else:
                derivatives[episode_no] = 1 / (self.action_prob - 1)

        return derivatives

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