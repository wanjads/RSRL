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


def plot_loss(losses):

    # calculate moving average
    averages = []
    length = constants.moving_average_length
    for i in range(len(losses) - length + 1):
        average = sum(losses[i:i + length]) / length
        averages.append(average)

    df = pd.DataFrame(dict(
        episode=range(constants.max_episodes - length + 1),
        loss=averages
    ))

    fig = px.line(df, x="episode", y="loss", title='moving average: loss over ' + str(length) + ' episodes')
    fig.show()


def main():

    losses = []
    episode_no = 0
    state = State.initial_state()
    strategy = Strategy()
    epsilon = 0.5
    decay = 0.999

    while not stop(episode_no):

        action = strategy.action(state, epsilon)
        state, cost = state.update(action)

        episode_no += 1
        epsilon = decay * epsilon

        loss = strategy.update(state, action, cost, episode_no)
        losses += [loss]

        if episode_no / 100 == 0:
            print(str(episode_no/constants.max_episodes * 100) + " %")

    plot_loss(losses)


if __name__ == '__main__':
    main()
