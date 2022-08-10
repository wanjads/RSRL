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
        self.nn = network.NN(3)

        # these strategies need extra information
        if strategy_type == "mean_variance" or strategy_type == "semi_std_deviation":
            self.mean = 0
        if strategy_type == "cvar":
            self.sorted_costs = []
        if strategy_type == "fishburn":
            self.target = constants.energy_weight + 1
        if strategy_type == "risk_monte_carlo":
            self.penalties = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, 2))
        if strategy_type == "REINFORCE_action_prob":
            self.action_prob = 0.6
        if strategy_type == "REINFORCE_sigmoid":
            self.flat = 5.3219222624013645
            self.shift = 8.352662521479429
        if strategy_type == "tabular_Q":
            self.qvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, 2))
            self.learning_rate = 0.01
        if strategy_type == "value_iteration":
            self.qvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, 2))
            self.value_iteration()

    # the tabular q-learning update dependent on risk sensitivity
    def update(self, old_state, state, action, episode_no):

        cost = constants.energy_weight * action + state.aoi_receiver
        inp = old_state.as_input()

        if self.strategy_type == "utility_function":  # see shen et al. p.9 eq (13) for tabular version
            utility = utils.utility_function(cost, self.risk_factor)
            self.nn.train_model(inp, action, utility)

        # zhou et al. only use the cost from aoi but in this context using the general cost makes more sense
        elif self.strategy_type == "cvar":  # see zhou et al.
            bisect.insort(self.sorted_costs, cost)
            risk = utils.cvar_risk(self.sorted_costs, self.risk_factor)
            cvar_cost = cost + 1 * risk  # risk factor mu seems to be irrelevant
            self.nn.train_model(inp, action, cvar_cost)

        elif self.strategy_type == "mean_variance":  # own idea
            self.mean = utils.running_mean(episode_no, self.mean, cost)
            mv_cost = cost + self.risk_factor * (cost - self.mean) ** 2
            self.nn.train_model(inp, action, mv_cost)

        elif self.strategy_type == "fishburn":  # own idea
            if cost < self.target:
                stone_cost = cost
            else:
                stone_cost = cost + self.risk_factor * (cost - self.target)
            self.nn.train_model(inp, action, stone_cost)

        elif self.strategy_type == "semi_std_deviation":  # own idea
            self.mean = utils.running_mean(episode_no, self.mean, cost)
            if cost < self.mean:
                sd_cost = cost
            else:
                sd_cost = cost + self.risk_factor * (cost - self.mean)
            self.nn.train_model(inp, action, sd_cost)

        elif self.strategy_type == "network_Q" or self.strategy_type == "stochastic":  # standard q-learning
            self.nn.train_model(inp, action, cost)

        elif self.strategy_type == "risk_states":  # own idea
            risky_state_cost = cost
            if state.aoi_receiver >= constants.risky_aoi:
                risky_state_cost = self.risk_factor * cost
            self.nn.train_model(inp, action, risky_state_cost)

        elif self.strategy_type == "tabular_Q":  # tabular q-learning
            V = np.min(self.qvalues[state.aoi_sender][state.aoi_receiver])
            old_q_value = self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][action]
            self.qvalues[old_state.aoi_sender][old_state.aoi_receiver][action] = \
                (1 - self.learning_rate) * old_q_value + self.learning_rate * (cost + constants.gamma * V)

        else:
            print("a strategy update for strategy type " + self.strategy_type + " is not implemented")

    def update_reinforce(self, states, actions, costs):

        if self.strategy_type == "REINFORCE_action_prob":
            returns = self.returns_from_costs(costs)
            derivatives = self.derivatives_from_state_actions(states, actions)

            for episode_no in range(len(states)):
                self.action_prob = \
                    max(0, min(1, self.action_prob + 0.00000005 * derivatives[episode_no] * returns[episode_no]))

        elif self.strategy_type == "REINFORCE_sigmoid":
            returns = self.returns_from_costs(costs)
            derivatives = self.derivatives_sigmoid_strategy(states, actions)

            for episode_no in range(len(states)):
                self.flat += 0.000001 * derivatives[episode_no][0] * returns[episode_no]
                self.shift += 0.00001 * derivatives[episode_no][1] * returns[episode_no]

    def update_risk_monte_carlo(self, state, action, costs):
        self.penalties[state.aoi_sender][state.aoi_receiver][action] += sum(costs)

    def action(self, state, epsilon):
        if self.strategy_type == 'always':
            action = 1
        elif self.strategy_type == 'never':
            action = 0
        elif self.strategy_type == 'random':
            action = random.randint(0, 1)
        elif self.strategy_type == 'send_once':
            action = int(state.aoi_sender == 0)
        elif self.strategy_type == 'benchmark2':
            action = 0
            if constants.send_prob * (state.aoi_receiver - state.aoi_sender - 1) \
                    * (1 + max(0, constants.energy_weight - state.aoi_receiver)
                       + constants.new_package_prob / (1 - constants.new_package_prob) ** 2) \
                    >= constants.energy_weight:
                action = 1
        elif self.strategy_type == 'optimal_threshold':
            action = 0
            # this is tuned for lambda = 0.9, p = 0.5, e = 3
            if state.aoi_receiver - state.aoi_sender >= 2:
                action = 1
        elif self.strategy_type == 'threshold':
            action = 0
            # this is tuned for lambda = 0.9, p = 0.5, e = 3
            if state.aoi_receiver - state.aoi_sender >= 3:
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
        elif self.strategy_type == 'risk_monte_carlo':
            action = 0
            if self.penalties[state.aoi_sender][state.aoi_receiver][0] == \
                    self.penalties[state.aoi_sender][state.aoi_receiver][1]:
                action = random.randint(0, 1)
            elif self.penalties[state.aoi_sender][state.aoi_receiver][0] > \
                    self.penalties[state.aoi_sender][state.aoi_receiver][1]:
                action = 1
        elif self.strategy_type == 'REINFORCE_action_prob':
            action_probs = [(1 - self.action_prob), self.action_prob]
            action = np.random.choice([0, 1], p=action_probs)
        elif self.strategy_type == 'REINFORCE_sigmoid':
            x = state.aoi_receiver - state.aoi_sender
            action_probs = [(1 - utils.sigmoid(self.flat * x - self.shift)),
                            utils.sigmoid(self.flat * x - self.shift)]
            action = np.random.choice([0, 1], p=action_probs)
        elif random.random() < epsilon:
            action = random.randint(0, 1)
        elif self.strategy_type == 'tabular_Q' or self.strategy_type == 'value_iteration':
            if self.qvalues[state.aoi_sender][state.aoi_receiver][0] == \
                    self.qvalues[state.aoi_sender][state.aoi_receiver][1]:
                action = random.randint(0, 1)
            else:
                action = np.argmin(self.qvalues[state.aoi_sender][state.aoi_receiver])
        elif self.strategy_type == 'stochastic':
            action = 0
            out = self.nn.out(state.as_input())[0]
            if random.random() < out[1] / out[0] + out[1]:
                action = 1
        else:
            action = np.argmin(self.nn.out(state.as_input())[0])

        return action

    def simulate(self, length, state, simulation_type, action):

        state = copy.deepcopy(state)
        data = []
        for simulation_episode in range(length):
            state.update(action)
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

    def derivatives_sigmoid_strategy(self, states, actions):

        derivatives = np.zeros((constants.reinforce_rollout_length, 2))

        for episode_no in range(len(states)):
            x = states[episode_no][1] - states[episode_no][0]
            if actions[episode_no]:
                derivatives[episode_no][0] = (1 - utils.sigmoid(self.flat * x - self.shift)) * x
                derivatives[episode_no][1] = utils.sigmoid(self.flat * x - self.shift) - 1
            else:
                derivatives[episode_no][0] = - utils.sigmoid(self.flat * x - self.shift) * x
                derivatives[episode_no][1] = utils.sigmoid(self.flat * x - self.shift)

        return derivatives

    def value_iteration(self):

        vvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1))
        qvalues = np.zeros((constants.aoi_cap + 1, constants.aoi_cap + 1, 2))

        eps = 5
        aoi_cap = 10
        for iteration in range(eps):
            for aois in range(aoi_cap + 1):
                print(aois + 100 * iteration)
                for aoir in range(aoi_cap + 1):
                    for action in range(2):
                        summe = 0
                        p_summe = 0
                        for aois2 in range(aoi_cap + 1):
                            for aoir2 in range(aoi_cap + 1):
                                if not action:
                                    p = constants.new_package_prob * \
                                        (aois2 == 0 and (aoir2 == aoir + 1 or aoir2 == aoi_cap == aoir)) \
                                        + (1 - constants.new_package_prob) * \
                                        ((aois2 == aois + 1 or aois2 == aoi_cap == aois) and
                                         (aoir2 == aoir + 1 or aoir2 == aoi_cap == aoir))
                                    r = aoir2
                                else:
                                    p = constants.new_package_prob * constants.send_prob * \
                                        (aois2 == 0 and (aoir2 == aois + 1 or aoir2 == aoi_cap == aois)) \
                                        + constants.new_package_prob * (1 - constants.send_prob) * \
                                        (aois2 == 0 and (aoir2 == aoir + 1 or aoir2 == aoi_cap == aoir)) \
                                        + (1 - constants.new_package_prob) * constants.send_prob * \
                                        ((aois2 == aois + 1 or aois2 == aoi_cap == aois) and
                                         (aoir2 == aois + 1 or aoir2 == aoi_cap == aois)) \
                                        + (1 - constants.new_package_prob) * (1 - constants.send_prob) * \
                                        ((aois2 == aois + 1 or aois2 == aoi_cap == aois) and
                                         (aoir2 == aoir + 1 or aoir2 == aoi_cap == aoir))
                                    r = aoir2 + constants.energy_weight
                                summe += p * (r + constants.gamma * vvalues[aois2][aoir2])
                                p_summe += p
                        qvalues[aois][aoir][action] = summe
                    vvalues[aois][aoir] = np.min(qvalues[aois][aoir])

        for aois in range(constants.aoi_cap + 1):
            for aoir in range(constants.aoi_cap + 1):
                self.qvalues[aois][aoir][0] = qvalues[aois][aoir][0]
                self.qvalues[aois][aoir][1] = qvalues[aois][aoir][1]
