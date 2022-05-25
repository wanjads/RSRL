# imports
from state import State
from strategy import Strategy
import constants
import os
import plotly.express as px
import pandas as pd

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

    losses = []

    state = State.initial_state()
    strategy = Strategy(risk)

    epsilon = constants.epsilon_0
    for episode_no in range(constants.train_episodes):

        action = strategy.action(state, epsilon)
        state, cost = state.update(action)

        if strategy.risk_sensitivity:
            cost = constants.risk_function(cost)

        epsilon = constants.decay * epsilon

        loss = strategy.update(state, action, cost, episode_no + 1)
        losses += [loss]

        if episode_no % 100 == 0:
            print(str(int(episode_no / constants.train_episodes * 100)) + " %")

    plot_moving_avg(losses, 'loss', constants.train_episodes, risk)

    print("---------- TRAINING COMPLETE ----------")

    return strategy


def test(strategy):
    print("----------   TEST STRATEGY   ----------")
    print("risk sensitivity: " + str(strategy.risk_sensitivity))

    costs = []
    risk_weighted_costs = []

    state = State.initial_state()
    for episode_no in range(constants.test_episodes):

        action = strategy.action(state, 0)
        state, cost = state.update(action)

        costs += [cost]
        risk_weighted_costs += [constants.risk_function(cost)]

        if episode_no % 100 == 0:
            print(str(int(episode_no / constants.test_episodes * 100)) + " %")

    print("avg cost: " + str(sum(costs) / len(costs)))
    print("avg risk weighted cost: " + str(sum(risk_weighted_costs) / len(risk_weighted_costs)))

    plot_moving_avg(costs, 'cost', constants.test_episodes, strategy.risk_sensitivity)
    plot_moving_avg(risk_weighted_costs, 'risk weighted cost', constants.test_episodes, strategy.risk_sensitivity)

    print("----------   TEST COMPLETE   ----------")


def main():
    risk_neutral_strategy = train(False)
    risk_sensitive_strategy = train(True)

    test(risk_neutral_strategy)
    test(risk_sensitive_strategy)


if __name__ == '__main__':
    main()
