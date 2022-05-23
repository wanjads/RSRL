# imports
from state import State
from strategy import strategy
import constants


def stop(n):
    if n >= constants.max_episodes:
        return True
    return False


def main():
    cumulative_cost = 0
    episode_no = 0
    state = State.initial_state()
    while not stop(episode_no):
        action = strategy(state)
        state, cost = state.update(action)
        cumulative_cost += cost
        episode_no += 1
        print(cumulative_cost/episode_no)


if __name__ == '__main__':
    main()
