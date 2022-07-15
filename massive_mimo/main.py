from map import Map
from strategy import Strategy
import utils
import random
import numpy as np
from state import State
import constants
import copy


def train(strategy_type, m, risk_factor):
    print("----------    TRAIN MODEL    ----------")
    print("strategy type: " + strategy_type)

    initial_action = 1 / constants.K * np.ones(constants.K)
    state = State.initial_state(initial_action, m)
    strategy = Strategy(strategy_type, risk_factor, m)

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


def test(strategy, m, data):
    print("----------   TEST STRATEGY   ----------")
    print("strategy type: " + str(strategy.strategy_type))

    costs = []
    initial_action = 1/constants.K * np.ones(constants.K)
    state = State.initial_state(initial_action, m)

    for episode_no in range(constants.test_episodes):

        action = strategy.action(state, 0)

        state.update(action)

        cost = utils.cost(state)
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

    # random seeds
    random.seed(10)
    np.random.seed(10)

    # init map m
    m = Map(1000)

    # init/train strategies
    strategy = train("TEST", m, 0)

    # test strategies
    data = {'strategy': [], 'avg_cost': [], 'risk': []}
    test(strategy, m, data)

    # TEST TEST TEST
    s = utils.sinr(0, m, strategy.action("", ""))
    print(s)


if __name__ == '__main__':
    main()
