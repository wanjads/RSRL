import math
import numpy as np
import constants
import random


class Map:

    def __init__(self, size):

        # size of the quadratic map in meters
        self.size = size

        # list of the coordinates of all UEs
        self.ues = []

        # list of the coordinates of all ACs
        self.acs = []

        for _ in range(constants.K):
            self.ues += [(size * random.random(), size * random.random())]

        for _ in range(constants.M):
            self.acs += [(size * random.random(), size * random.random())]

        self.distances = np.zeros((constants.M, constants.K))

        for ac in range(constants.M):
            for ue in range(constants.K):
                self.distances[ac][ue] = \
                    math.sqrt((self.acs[ac][0] - self.ues[ue][0])**2 + (self.acs[ac][1] - self.ues[ue][1])**2)

        self.beta = np.zeros((constants.M, constants.K))

        for ac in range(constants.M):
            for ue in range(constants.K):
                self.beta[ac][ue] = (constants.c / (4 * constants.f * math.pi))**2 * 1 / \
                                    self.distances[ac][ue]**constants.alpha

        self.gamma = np.zeros((constants.M, constants.K))

        for ac in range(constants.M):
            for ue in range(constants.K):
                self.gamma[ac][ue] = (constants.tau_p * constants.rho_u * self.beta[ac][ue]**2) / \
                                     (constants.tau_p * constants.rho_u
                                      * constants.pilot_sequence_constant * np.sum(self.beta[ac]) + 1)
