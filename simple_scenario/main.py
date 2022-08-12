import numpy as np
from state import State
from strategy import Strategy
import constants
import copy
import random
import utils
import os

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# train a strategy using (risk-sens.) eps-greedy q-learning
def train(strategy_type, risk_factor):
    print("----------    TRAIN MODEL    ----------")
    print("strategy type: " + strategy_type)

    state = State.initial_state()
    strategy = Strategy(strategy_type, risk_factor)

    epsilon = constants.epsilon_0
    for episode_no in range(constants.train_episodes):

        old_state = copy.deepcopy(state)
        action = strategy.action(state, epsilon)
        state.update(action)

        epsilon = constants.decay * epsilon

        strategy.update(old_state, state, action, episode_no)

        if episode_no % int(0.2 * constants.train_episodes) == 0:
            print(str(int(episode_no / constants.train_episodes * 100)) + " %")

    print("100 %")
    print("---------- TRAINING COMPLETE ----------")
    print()

    return strategy


def train_reinforce(strategy_type, risk_factor):

    # REINFORCE ASSUMES KNOWLEDGE OF THE RELEVANT PROBABILITIES

    print("----------    TRAIN MODEL    ----------")
    print("strategy type: " + strategy_type)

    strategy = Strategy(strategy_type, risk_factor)
    parameters = [[], []]

    for trajectory_no in range(constants.no_train_trajectories):

        state = State.initial_state()
        costs = np.zeros(constants.reinforce_rollout_length)
        actions = np.zeros(constants.reinforce_rollout_length)
        states = np.zeros((constants.reinforce_rollout_length, 3))

        for episode_no in range(constants.reinforce_rollout_length):

            states[episode_no] = [state.aoi_sender, state.aoi_receiver, state.last_action]
            action = strategy.action(state, 0)
            actions[episode_no] = action
            state.update(action)

            costs[episode_no] = state.aoi_receiver + constants.energy_weight * action

        strategy.update_reinforce(states, actions, costs)

        if strategy_type == "REINFORCE_sigmoid":
            parameters[0] += [strategy.flat]
            parameters[1] += [strategy.shift]
        elif strategy_type == "REINFORCE_action_prob":
            parameters[0] += [strategy.action_prob]

        if trajectory_no % int(0.2 * constants.no_train_trajectories) == 0:
            print(str(int(trajectory_no / constants.no_train_trajectories * 100)) + " %")

    if strategy_type == "REINFORCE_sigmoid":
        utils.line_plot(parameters[0])
        utils.line_plot(parameters[1])
    elif strategy_type == "REINFORCE_action_prob":
        utils.line_plot(parameters[0])

    print("100 %")
    print("---------- TRAINING COMPLETE ----------")
    print()

    return strategy


def train_risk_monte_carlo(strategy_type, risk_factor):

    print("----------    TRAIN MODEL    ----------")
    print("strategy type: " + strategy_type)

    strategy = Strategy(strategy_type, risk_factor)
    initial_aoi = [0, 0]

    for trajectory_no in range(constants.risk_monte_carlo_trajectories):

        state = State(initial_aoi[0], initial_aoi[1], 0)
        costs = np.zeros(constants.risk_monte_carlo_rollout_length)

        for episode_no in range(constants.risk_monte_carlo_rollout_length):

            if episode_no > 0:
                action = strategy.action(state, 0)
            else:
                action = trajectory_no % 2
            state.update(action)

            costs[episode_no] = state.aoi_receiver + constants.energy_weight * action

        strategy.update_risk_monte_carlo(State(initial_aoi[0], initial_aoi[1], 0), trajectory_no % 2, costs)

        if trajectory_no % 2 == 1:
            initial_aoi[0] += 1
            if initial_aoi[0] > constants.aoi_cap:
                initial_aoi[0] = 0
                initial_aoi[1] += 1
            if initial_aoi[1] > constants.aoi_cap:
                initial_aoi[0] = 0
                initial_aoi[1] = 0

        if trajectory_no % int(0.2 * constants.risk_monte_carlo_trajectories) == 0:
            print(str(int(trajectory_no / constants.risk_monte_carlo_trajectories * 100)) + " %")

    print("100 %")
    print("---------- TRAINING COMPLETE ----------")
    print()

    return strategy


# test a strategy calculating avg costs and risk
def test(strategy, data, run, no_of_runs):
    print("----------   TEST STRATEGY   ----------")
    print("strategy type: " + str(strategy.strategy_type))

    costs = []
    risky_states = 0
    state = State.initial_state()
    for episode_no in range(constants.test_episodes):

        action = strategy.action(state, 0)

        state.update(action)

        cost = constants.energy_weight * action + state.aoi_receiver

        costs += [cost]

        if state.aoi_receiver >= constants.risky_aoi:
            risky_states += 1/constants.test_episodes

        if episode_no % int(0.2 * constants.test_episodes) == 0:
            print(str(int(episode_no / constants.test_episodes * 100)) + " %")

    print("100 %")

    avg_cost = sum(costs) / len(costs)
    risk = utils.semi_std_dev(costs)
    fishburn_risk = utils.fishburn_measure(costs, constants.energy_weight + 1)
    print("avg cost: " + str(avg_cost))
    print("risk: " + str(risk))
    print("risky states: " + str(risky_states))
    print("fishburn's measure: " + str(fishburn_risk))

    print("----------   TEST COMPLETE   ----------")
    print()

    if run == 0:
        data['strategy'] += [strategy.strategy_type]
        data['avg_cost'] += [avg_cost/no_of_runs]
        data['risk'] += [risk/no_of_runs]
        data['risky_states'] += [risky_states/no_of_runs]
        data['fishburn'] += [fishburn_risk/no_of_runs]
    else:
        index = data['strategy'].index(strategy.strategy_type)
        data['avg_cost'][index] += avg_cost/no_of_runs
        data['risk'][index] += risk/no_of_runs
        data['risky_states'][index] += risky_states/no_of_runs
        data['fishburn'][index] += fishburn_risk/no_of_runs


def risk_factor_train_test(strategy, step, data):
    strategies = []
    for risk in range(10):
        strategies += [train(strategy, step * risk)]
    for strategy in strategies:
        test(strategy, data)
    utils.risk_factor_cost_bar_chart(data, step, strategy.strategy_type)
    utils.risk_factor_risk_bar_chart(data, step, strategy.strategy_type)


def main():

    # for risk_factor in [0.045, 0.0475, 0.05, 0.0525, 0.055]:
    # print("risk_factor: " + str(risk_factor))
    no_of_runs = 5
    data = {'strategy': [], 'avg_cost': [], 'risk': [], 'risky_states': [], 'fishburn': []}
    for random_seed in range(no_of_runs):

        # set a random seed for reproducibility
        random.seed(random_seed)

        # init two benchmark strategy sending never / in every episode
        # always_strategy = Strategy("always", 0)
        # never_strategy = Strategy("never", 0)

        # init a strategy acting uniformly random
        random_strategy = Strategy("random", 0)

        # init a benchmark sending, if a new package arrived
        # benchmark_strategy = Strategy("send_once", 0)
        # init a more sophisticated benchmark
        # benchmark2_strategy = Strategy("benchmark2", 0)
        # init the threshold strategy
        # threshold_strategy = Strategy("threshold", 0)
        # init the optimal threshold strategy
        optimal_strategy = Strategy("optimal_threshold", 0)

        # train a risk neutral strategy and risk averse strategies in different variants
        risk_neutral_strategy = train("network_Q", 0)
        # stochastic_risk_neutral_strategy = train("stochastic", 0)
        variance_strategy = train("mean_variance", 0.25)
        semi_std_dev_strategy = train("semi_std_deviation", 0.5)
        fishburn_strategy = train("fishburn", 0.5)
        cvar_strategy = train("cvar", 0.5)
        utility_strategy = train("utility_function", 0.0475)
        risk_states_strategy = train("risk_states", 2)
        # basic_monte_carlo_strategy = Strategy("basic_monte_carlo", 0)
        # old_train_eps = constants.train_episodes
        # constants.train_episodes = 1000000
        # tabular_strategy = train("tabular_Q", 0)
        # constants.train_episodes = old_train_eps
        # value_iteration = Strategy("value_iteration", 0)

        # reinforce_strategy_action_prob = train_reinforce("REINFORCE_action_prob", 0)

        #TODO die gelernte sigmoid strategie ist irgendwie besser
        # als die selbe ohne vorheriges lernen mit denselben parametern

        # reinforce_strategy_sigmoid = train_reinforce("REINFORCE_sigmoid", 0)
        # reinforce_strategy_action_prob = Strategy("REINFORCE_action_prob", 0)
        # reinforce_strategy_sigmoid = Strategy("REINFORCE_sigmoid", 0)
        # risk_monte_carlo_strategy = train_risk_monte_carlo("risk_monte_carlo", 0)

        # test all strategies
        # data collects all costs and risks
        # test(always_strategy, data, random_seed, no_of_runs)
        # test(never_strategy, data, random_seed, no_of_runs)
        test(random_strategy, data, random_seed, no_of_runs)
        # test(benchmark_strategy, data, random_seed, no_of_runs)
        # test(benchmark2_strategy, data, random_seed, no_of_runs)
        # test(threshold_strategy, data, random_seed, no_of_runs)
        test(optimal_strategy, data, random_seed, no_of_runs)
        test(risk_neutral_strategy, data, random_seed, no_of_runs)
        # test(stochastic_risk_neutral_strategy, data, random_seed, no_of_runs)
        test(variance_strategy, data, random_seed, no_of_runs)
        test(semi_std_dev_strategy, data, random_seed, no_of_runs)
        test(fishburn_strategy, data, random_seed, no_of_runs)
        test(cvar_strategy, data, random_seed, no_of_runs)
        test(utility_strategy, data, random_seed, no_of_runs)
        test(risk_states_strategy, data, random_seed, no_of_runs)
        # test(basic_monte_carlo_strategy, data, random_seed, no_of_runs)
        # test(reinforce_strategy_action_prob, data, random_seed, no_of_runs)
        # test(reinforce_strategy_sigmoid, data, random_seed, no_of_runs)
        # test(risk_monte_carlo_strategy, data, random_seed, no_of_runs)
        # test(tabular_strategy, data, random_seed, no_of_runs)
        # test(value_iteration, data, random_seed, no_of_runs)

    # round values in data
    for i in range(1, len(data)):
        data[list(data.keys())[i]] = list(map(lambda x: round(x, 4), data[list(data.keys())[i]]))

    # plot bar charts
    utils.bar_chart(data, 'avg_cost', True)
    utils.bar_chart(data, 'risk', False)
    utils.bar_chart(data, 'risky_states', False)
    utils.bar_chart(data, 'fishburn', False)


if __name__ == '__main__':
    main()
