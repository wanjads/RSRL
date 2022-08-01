from state import State
from strategy import Strategy
import constants
import copy
import random
import utils


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
        state = state.update(action)

        epsilon = constants.decay * epsilon

        strategy.update(old_state, state, action, utils.learning_rate(episode_no), episode_no)

        if episode_no % int(0.2 * constants.train_episodes) == 0:
            print(str(int(episode_no / constants.train_episodes * 100)) + " %")

    print("100 %")
    print("---------- TRAINING COMPLETE ----------")
    print()

    return strategy


# test a strategy calculating avg costs and risk
def test(strategy, data):
    print("----------   TEST STRATEGY   ----------")
    print("strategy type: " + str(strategy.strategy_type))

    costs = []

    state = State.initial_state()
    for episode_no in range(constants.test_episodes):

        action = strategy.action(state, 0)

        state = state.update(action)

        cost = constants.energy_weight * action + state.aoi_receiver
        costs += [cost]

        if episode_no % int(0.2 * constants.test_episodes) == 0:
            print(str(int(episode_no / constants.test_episodes * 100)) + " %")

    print("100 %")

    avg_cost = sum(costs) / len(costs)
    risk = utils.semi_std_dev(costs)
    print("avg cost: " + str(avg_cost))
    print("risk: " + str(risk))

    print("----------   TEST COMPLETE   ----------")
    print()

    data['strategy'] += [strategy.strategy_type]
    data['avg_cost'] += [avg_cost]
    data['risk'] += [risk]


def risk_factor_train_test(strategy, step):
    strategies = []
    for risk in range(10):
        strategies += [train(strategy, step * risk)]
    data = {'strategy': [], 'avg_cost': [], 'risk': []}
    for strategy in strategies:
        test(strategy, data)
    utils.risk_factor_cost_bar_chart(data, step, strategy.strategy_type)
    utils.risk_factor_risk_bar_chart(data, step, strategy.strategy_type)


def main():

    # set a random seed for reproducibility
    random.seed(10)

    # init two benchmark strategy sending never / in every episode
    always_strategy = Strategy("always", 0)
    # never_strategy = Strategy("never", 0)

    # init a benchmark sending, if a new package arrived
    benchmark_strategy = Strategy("benchmark", 0)

    # value iteration
    # value_iteration = Strategy("value_iteration", 0)

    # train a risk neutral strategy and risk averse strategies in different variants
    risk_neutral_strategy = train("risk_neutral", 0)
    risk_neutral_stochastic_strategy = train("stochastic", 0)
    # variance_strategy = train("mean_variance", 0.3)
    # semi_std_dev_strategy = train("semi_std_deviation", 0.1)
    # stone_strategy = train("stone_measure", 0.1)
    # cvar_strategy = train("cvar", 0.05)
    # utility_strategy = train("utility_function", 0.05)

    # test all strategies
    # data collects all costs and risks
    data = {'strategy': [], 'avg_cost': [], 'risk': []}
    test(always_strategy, data)
    # test(never_strategy, data)
    test(benchmark_strategy, data)
    test(risk_neutral_strategy, data)
    test(risk_neutral_stochastic_strategy, data)
    # test(variance_strategy, data)
    # test(semi_std_dev_strategy, data)
    # test(stone_strategy, data)
    # test(cvar_strategy, data)
    # test(utility_strategy, data)
    # test(value_iteration, data)

    # plot bar charts
    utils.bar_chart(data, 'avg_cost', True)
    utils.bar_chart(data, 'risk', False)


if __name__ == '__main__':
    main()
