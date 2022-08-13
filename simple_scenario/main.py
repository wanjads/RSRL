import numpy as np
from state import State
from strategy import Strategy
import constants
import copy
import utils
import random
import os

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# suppress gpu usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# train a strategy using (risk-sens.) eps-greedy q-learning
def train(strategy_type, risk_factor, train_episodes):
    print("----------    TRAIN MODEL    ----------")
    print("strategy type: " + strategy_type)

    state = State.initial_state()
    strategy = Strategy(strategy_type, risk_factor)

    epsilon = constants.epsilon_0

    for episode_no in range(train_episodes):

        old_state = copy.deepcopy(state)
        action = strategy.action(state, epsilon)
        state.update(action)

        epsilon = constants.decay * epsilon

        strategy.update(old_state, state, action, episode_no)

        if episode_no % int(0.2 * train_episodes) == 0:
            print(str(int(episode_no / train_episodes * 100)) + " %")

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

    # if strategy_type == "REINFORCE_sigmoid":
    #     utils.line_plot(parameters[0])
    #     utils.line_plot(parameters[1])
    # elif strategy_type == "REINFORCE_action_prob":
    #     utils.line_plot(parameters[0])

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


def main():

    for experiment in ["tabular"]:  # "non-learning", "risk-neutral", "tabular", "network"

        print(experiment)

        no_of_runs = 10
        # data collects all costs and risks
        data = {'strategy': [], 'avg_cost': [], 'risk': [], 'risky_states': [], 'fishburn': []}
        for random_seed in range(no_of_runs):

            print("run_no: " + str(random_seed))
            # set a random seed for reproducibility
            random.seed(random_seed)

            # non-learning strategies
            if experiment == "non-learning":
                always = Strategy("always", 0)
                # never = Strategy("never", 0)
                rand = Strategy("random", 0)
                send_once = Strategy("send_once", 0)
                threshold = Strategy("threshold", 0)
                optimal_threshold = Strategy("optimal_threshold", 0)
                basic_monte_carlo = Strategy("basic_monte_carlo", 0)

                # test strategies
                test(always, data, random_seed, no_of_runs)
                # test(never, data, random_seed, no_of_runs)
                test(rand, data, random_seed, no_of_runs)
                test(send_once, data, random_seed, no_of_runs)
                test(threshold, data, random_seed, no_of_runs)
                test(optimal_threshold, data, random_seed, no_of_runs)
                test(basic_monte_carlo, data, random_seed, no_of_runs)

            # risk-neutral learning strategies
            if experiment == "risk-neutral":
                rand = Strategy("random", 0)
                optimal_threshold = Strategy("optimal_threshold", 0)
                # value_iteration = Strategy("value_iteration", 0)
                tabular_Q = train("tabular_Q", 0, 100000)
                network_Q = train("network_Q", 0, 1000)
                reinforce_action_prob = train_reinforce("REINFORCE_action_prob", 0)
                reinforce_sigmoid = train_reinforce("REINFORCE_sigmoid", 0)

                # test strategies
                test(rand, data, random_seed, no_of_runs)
                test(optimal_threshold, data, random_seed, no_of_runs)
                # test(value_iteration, data, random_seed, no_of_runs)
                test(tabular_Q, data, random_seed, no_of_runs)
                test(network_Q, data, random_seed, no_of_runs)
                test(reinforce_action_prob, data, random_seed, no_of_runs)
                test(reinforce_sigmoid, data, random_seed, no_of_runs)

            # risk-sensitive tabular strategies
            if experiment == "tabular":
                rand = Strategy("random", 0)
                optimal_threshold = Strategy("optimal_threshold", 0)
                tabular_Q = train("tabular_Q", 0, 1000000)
                mean_variance_tabular = train("mean_variance_tabular", 0.25, 1000000)
                semi_std_dev_tabular = train("semi_std_deviation_tabular", 0.5, 1000000)
                fishburn_tabular = train("fishburn_tabular", 0.5, 1000000)
                cvar_tabular = train("cvar_tabular", 0.5, 1000000)
                utility_tabular = train("utility_function_tabular", 0.475, 1000000)
                risk_states_tabular = train("risk_states_tabular", 2, 1000000)

                # test strategies
                test(rand, data, random_seed, no_of_runs)
                test(optimal_threshold, data, random_seed, no_of_runs)
                test(tabular_Q, data, random_seed, no_of_runs)
                test(mean_variance_tabular, data, random_seed, no_of_runs)
                test(semi_std_dev_tabular, data, random_seed, no_of_runs)
                test(fishburn_tabular, data, random_seed, no_of_runs)
                test(cvar_tabular, data, random_seed, no_of_runs)
                test(utility_tabular, data, random_seed, no_of_runs)
                test(risk_states_tabular, data, random_seed, no_of_runs)

            # risk-sensitive network based strategies
            if experiment == "network":
                rand = Strategy("random", 0)
                optimal_threshold = Strategy("optimal_threshold", 0)
                network_Q = train("network_Q", 0, 1000)
                mean_variance_network = train("mean_variance_network", 0.25, 1000)
                semi_std_dev_network = train("semi_std_deviation_network", 0.5, 1000)
                fishburn_network = train("fishburn_network", 0.5, 1000)
                cvar_network = train("cvar_network", 0.5, 1000)
                utility_network = train("utility_function_network", 0.0475, 1000)
                risk_states_network = train("risk_states_network", 2, 1000)

                # test strategies
                test(rand, data, random_seed, no_of_runs)
                test(optimal_threshold, data, random_seed, no_of_runs)
                test(network_Q, data, random_seed, no_of_runs)
                test(mean_variance_network, data, random_seed, no_of_runs)
                test(semi_std_dev_network, data, random_seed, no_of_runs)
                test(fishburn_network, data, random_seed, no_of_runs)
                test(cvar_network, data, random_seed, no_of_runs)
                test(utility_network, data, random_seed, no_of_runs)
                test(risk_states_network, data, random_seed, no_of_runs)

        # round values in data
        for i in range(1, len(data)):
            data[list(data.keys())[i]] = list(map(lambda x: round(x, 4), data[list(data.keys())[i]]))

        # plot bar charts
        utils.bar_chart(data, 'avg_cost', True, experiment)
        utils.bar_chart(data, 'risk', False, experiment)
        utils.bar_chart(data, 'risky_states', False, experiment)
        utils.bar_chart(data, 'fishburn', False, experiment)


if __name__ == '__main__':
    main()
