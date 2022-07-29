import constants
import numpy as np
import math


def sinr(k, m, action):

    s1 = np.sum((action**0.5 * m.gamma)[k])

    s2 = (np.sum(action**0.5 * m.gamma * 1/m.beta * np.transpose(np.tile(m.beta[:, 4], (5, 1))), axis=0))**2
    s2 = np.delete(s2, k)
    s2 = np.sum(s2)

    s3 = np.sum(np.sum(action * m.gamma * np.transpose(np.tile(m.beta[:, 4], (5, 1))), axis=0))

    return (s1**2) / (s2 * constants.pilot_sequence_constant + s3 + 1 / constants.rho_d)


def sinrs(m, action):

    s = np.zeros(constants.K)

    for k in range(constants.K):
        s[k] = sinr(k, m, action)

    return s


def reward(state):

    c = 0
    for k in range(constants.K):
        c += math.log(1 + state.sinrs[k], 2)

    return c


def semi_std_dev(costs):
    # semi standard deviation
    # see pedersen and satchell 1998

    mean = sum(costs) / len(costs)

    risk = 0
    for c in costs:
        if c > mean:
            risk += 1 / len(costs) * (c - mean) ** 2

    return math.sqrt(risk)
