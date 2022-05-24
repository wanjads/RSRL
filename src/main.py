# imports
from state import State
from strategy import Strategy
import constants


def stop(n):
    if n >= constants.max_episodes:
        return True
    return False


def main():
    cumulative_cost = 0
    episode_no = 0
    state = State.initial_state()
    strategy = Strategy()
    while not stop(episode_no):
        action = strategy.action(state, episode_no)
        state, cost = state.update(action)
        cumulative_cost += cost
        episode_no += 1
        strategy.update(state, action, episode_no)
        # print(cumulative_cost/episode_no)


if __name__ == '__main__':
    main()
