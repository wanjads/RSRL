def stop(n):
    if n >= 100:
        return True
    return False


def get_initial_state():
    return 0


def strategy(state):
    return 0


def update_state(state, action):
    return state


def main():
    episode_no = 0
    state = get_initial_state()
    while not stop(episode_no):
        action = strategy(state)
        state = update_state(state, action)
        episode_no += 1


if __name__ == '__main__':
    main()
