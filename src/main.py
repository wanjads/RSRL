# imports
from state import State
from strategy import Strategy
import constants
import plotly.express as px
import pandas as pd
import copy
import random


def plot_moving_avg(data, title, episodes, risk_sensitivity):
    # calculate moving average
    averages = []
    length = constants.moving_average_length
    for i in range(len(data) - length + 1):
        average = sum(data[i:i + length]) / length
        averages.append(average)

    df = pd.DataFrame({'episode': range(episodes - length + 1), title: averages})

    fig = px.line(df, x='episode', y=title, title='moving average: ' + title + ' over ' + str(length) + ' episodes.'
                                                  + ' risk sensitivity: ' + str(risk_sensitivity)
                                                  + '\t \t' + 'alpha: ' + str(constants.alpha)
                                                  + ', p: ' + str(constants.new_package_prob)
                                                  + ', lambda: ' + str(constants.send_prob)
                                                  + ', energy weight: ' + str(constants.energy_weight))
    fig.show()


def train(risk):
    print("----------    TRAIN MODEL    ----------")
    print("risk sensitivity: " + str(risk))

    state = State.initial_state()
    strategy = Strategy(risk)

    epsilon = constants.epsilon_0
    for episode_no in range(constants.train_episodes):

        old_state = copy.deepcopy(state)
        action = strategy.action(state, epsilon)
        state = state.update(action)

        epsilon = constants.decay * epsilon

        strategy.update(old_state, state, action, constants.learning_rate(episode_no))

        if episode_no % int(0.2 * constants.train_episodes) == 0:
            print(str(int(episode_no / constants.train_episodes * 100)) + " %")

    print("100 %")
    print("---------- TRAINING COMPLETE ----------")
    print()

    return strategy


def test(strategy):
    print("----------   TEST STRATEGY   ----------")
    print("risk sensitivity: " + str(strategy.risk_sensitivity))
    print("constant strategy: " + str(strategy.constant))

    costs = []

    state = State.initial_state()
    for episode_no in range(constants.test_episodes):

        action = strategy.action(state, 0)

        state = state.update(action)

        cost = constants.energy_weight * action + state.aoi_receiver
        costs += [cost]

        if episode_no % int(0.2 * constants.test_episodes) == 0:
            print(str(int(episode_no / constants.test_episodes * 100)) + " %")

    risk = constants.risk_measure(costs)
    print("avg cost: " + str(sum(costs) / len(costs)))
    print("risk: " + str(risk))

    # plot_moving_avg(costs, 'cost', constants.test_episodes, strategy.risk_sensitivity)
    # plot_moving_avg(risk_weighted_costs, 'risk weighted cost', constants.test_episodes, strategy.risk_sensitivity)

    # print("---   Q-VALUES   ---")
    # for aois in range(constants.aoi_cap):
    #     for aoir in range(aois + 1, constants.aoi_cap):
    #         for la in range(2):
    #             if not (la == 0 and aois == 0 and aoir == 1):
    #                 print("state (" + str(aois) + "," + str(aoir) + ", " + str(la) + "): "
    #                       + str(strategy.qvalues[aois][aoir][la]))

    print("100 %")
    print("----------   TEST COMPLETE   ----------")
    print()


def main():

    random.seed(10)

    risk_neutral_strategy = train(False)
    risk_sensitive_strategy = train(True)

    constant_strategy = Strategy(False)
    constant_strategy.send_always()

    test(constant_strategy)
    test(risk_neutral_strategy)
    test(risk_sensitive_strategy)


if __name__ == '__main__':
    main()
