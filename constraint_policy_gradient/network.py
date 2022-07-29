import copy

import numpy as np


class NN:

    def __init__(self, layer_lengths):
        self.weights = self.init_weights(layer_lengths)

    @staticmethod
    def init_weights(layer_lengths):
        weights = []
        for i in range(len(layer_lengths) - 1):
            l1 = layer_lengths[i]
            l2 = layer_lengths[i + 1]
            weights += [np.random.rand(l1 + 1, l2)]

        return weights

    @staticmethod
    def activation(x):
        return np.maximum(0, x)

    @staticmethod
    def dactivation(x):
        if x > 0:
            return 1
        return 0

    def predict(self, x):
        assert self.weights[0].shape[0] == x.shape[0] + 1
        for layer_weight_matrix in self.weights:
            x = np.concatenate((x, np.array([1])))
            x = np.matmul(x, layer_weight_matrix)
            x = self.activation(x)

        return x

    # risk constrained policy gradient update
    # trajectories has to be an array of N trajectories
    # each trajectory has to be a triple of
    # an array of x values and
    # an array of actions of the same length
    # an array of rewards of the same length
    # see: Prashanth and Michael 2022
    def train(self, trajectories):

        lagrange_multiplier = self.get_initial_lagrange_multiplier()

        for trajectory in trajectories:
            G = self.get_G_estimates(trajectory)
            dG = self.get_dG_estimates(trajectory)
            dJ = self.get_dJ_estimates(trajectory)

    def get_initial_lagrange_multiplier(self):
        lagrange_multiplier = copy.deepcopy(self.weights)
        return np.maximum(np.minimum(lagrange_multiplier, 0), 0)

    @staticmethod
    def projection_operator(x):
        parameter_max = 1
        parameter_min = -1
        np.maximum(np.minimum(x, parameter_max), parameter_min)

    def get_G_estimates(self, trajectory):
        # test with G(theta) = sum(parameters), kappa = 1
        s = np.sum(self.weights)
        G_estimate = []
        for layer in range(len(self.weights)):
            shape = self.weights[layer].shape
            G_estimate += [s * np.ones(shape)]

        return G_estimate

    def get_dG_estimates(self, trajectory):
        dG_estimate = []
        for layer in range(len(self.weights)):
            shape = self.weights[layer].shape
            dG_estimate += [np.ones(shape)]

        return dG_estimate

    def get_dJ_estimates(self, trajectory):
        dJ_estimate = []
        rewards_to_go = self.rewards_to_go(trajectory)
        for layer in range(len(self.weights)):
            shape = self.weights[layer].shape
            s = np.zeros(shape)
            for t in range(len(trajectory[0])):
                s += rewards_to_go[t] * self.dlogstrat(trajectory, t, layer)
            dJ_estimate += [s]

        return dJ_estimate

    def dlogstrat(self, trajectory, timestep, layer):
        return np.ones(self.weights[layer].shape)

    @staticmethod
    def rewards_to_go(trajectory):
        rewards = trajectory[2]
        rewards_to_go = []
        for t in range(len(rewards)):
            rewards_to_go += [sum(rewards[t:])]
        return rewards_to_go
