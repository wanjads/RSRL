# imports
from state import State
from strategy import strategy
import constants


def stop(n):
    if n >= constants.max_episodes:
        return True
    return False


def main():
    episode_no = 0
    state = State.initial_state()
    while not stop(episode_no):
        print(str(state.aoi_sender) + " : " + str(state.aoi_receiver))
        action = strategy(state)
        state = state.update(action)
        episode_no += 1


if __name__ == '__main__':
    main()
