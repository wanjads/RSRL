# imports
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
def test(strategy):
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

    absolute_risk = utils.risk_measure_absolute(costs)
    relative_risk = utils.risk_measure_expectation(costs)
    print("avg cost: " + str(sum(costs) / len(costs)))
    print("absolute risk: " + str(absolute_risk))
    print("relative risk: " + str(relative_risk))

    print("100 %")
    print("----------   TEST COMPLETE   ----------")
    print()


def main():

    # set a random seed for reproducibility
    random.seed(10)

    # init two benchmark strategy sending never / in every episode
    always_strategy = Strategy("always")
    never_strategy = Strategy("never")

    # train a risk neutral strategy and risk averse strategies in different variants
    risk_neutral_strategy = train("risk_neutral")
    variance_strategy = train("mean_variance")
    semi_std_dev_strategy = train("semi_std_deviation")
    stone_strategy = train("stone_measure")
    cvar_strategy = train("cvar")
    utility_strategy = train("utility_function")

    # test all strategies
    test(always_strategy)
    test(never_strategy)
    test(risk_neutral_strategy)
    test(variance_strategy)
    test(semi_std_dev_strategy)
    test(stone_strategy)
    test(cvar_strategy)
    test(utility_strategy)


if __name__ == '__main__':
    main()
