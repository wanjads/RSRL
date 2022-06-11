from state import State
from strategy import Strategy
import constants
import copy
import random
import utils


# train a strategy using (risk-sens.) eps-greedy q-learning
def train(strategy_type):
    print("----------    TRAIN MODEL    ----------")
    print("strategy type: " + strategy_type)

    state = State.initial_state()
    strategy = Strategy(strategy_type)

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


def main():

    # set a random seed for reproducibility
    random.seed(10)

    # init two benchmark strategy sending never / in every episode
    always_strategy = Strategy("always")
    never_strategy = Strategy("never")

    # init a benchmark sending, if a new package arrived
    benchmark_strategy = Strategy("benchmark")

    # train a risk neutral strategy and risk averse strategies in different variants
    risk_neutral_strategy = train("risk_neutral")
    variance_strategy = train("mean_variance")
    semi_std_dev_strategy = train("semi_std_deviation")
    stone_strategy = train("stone_measure")
    cvar_strategy = train("cvar")
    utility_strategy = train("utility_function")

    # test all strategies
    # data collects all costs and risks
    data = {'strategy': [], 'avg_cost': [], 'absolute_risk': [], 'risk': []}
    test(always_strategy, data)
    test(never_strategy, data)
    test(benchmark_strategy, data)
    test(risk_neutral_strategy, data)
    test(variance_strategy, data)
    test(semi_std_dev_strategy, data)
    test(stone_strategy, data)
    test(cvar_strategy, data)
    test(utility_strategy, data)

    # plot bar charts
    utils.bar_chart(data, 'avg_cost')
    utils.bar_chart(data, 'risk')


if __name__ == '__main__':
    main()
