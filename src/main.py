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

    df = pd.DataFrame(dict(
        episode=[i for i in range(constants.max_episodes)],
        loss=losses
    ))

    fig = px.line(df, x="episode", y="loss", title='loss over episodes')
    fig.show()


def main():
    losses = []
    cumulative_cost = 0
    episode_no = 0
    state = State.initial_state()
    strategy = Strategy()
    while not stop(episode_no):
        action = strategy.action(state)
        state, cost = state.update(action)
        cumulative_cost += cost
        episode_no += 1
        loss = strategy.update(state, action, cost, episode_no)
        losses += [loss]
        # print(cumulative_cost/episode_no)

    plot_loss(losses)


if __name__ == '__main__':
    main()
