# imports
from state import State
from strategy import Strategy
import constants
import os
import plotly.express as px
import pandas as pd


# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def stop(n):
    if n >= constants.max_episodes:
        return True
    return False


def plot_moving_avg(data, title):

    # calculate moving average
    averages = []
    length = constants.moving_average_length
    for i in range(len(data) - length + 1):
        average = sum(data[i:i + length]) / length
        averages.append(average)

    df = pd.DataFrame(dict(
        episode=range(constants.max_episodes - length + 1),
        loss=averages
    ))

    fig = px.line(df, x="episode", y="loss", title='moving average: ' + title + ' over ' + str(length) + ' episodes')
    fig.show()


def main():

    losses = []
    costs = []
    risk_weighted_costs = []

    episode_no = 0
    state = State.initial_state()
    strategy = Strategy()

    epsilon = constants.epsilon_0
    while not stop(episode_no):

        action = strategy.action(state, epsilon)
        state, cost = state.update(action)

        costs += [cost]
        if constants.risk:
            cost = constants.risk_function(cost)
            risk_weighted_costs += [cost]
        else:
            risk_weighted_costs += [constants.risk_function(cost)]

        episode_no += 1
        epsilon = constants.decay * epsilon

        loss = strategy.update(state, action, cost, episode_no)
        losses += [loss]

        if episode_no % 100 == 0:
            print(str(episode_no/constants.max_episodes * 100) + " %")

    plot_moving_avg(losses, 'loss')
    plot_moving_avg(costs, 'cost')
    plot_moving_avg(risk_weighted_costs, 'risk weighted cost')


if __name__ == '__main__':
    main()
