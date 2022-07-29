import numpy as np
import constants


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

    def predict(self, x):
        assert self.weights[0].shape[0] == x.shape[0] + 1
        for layer_weight_matrix in self.weights:
            x = np.concatenate((x, np.array([1])))
            x = np.matmul(x, layer_weight_matrix)
            x = self.activation(x)

        return x

    # policy gradient update
    # trajectories has to be an array of N trajectories
    # each trajectory has to be a triple of
    # an array of x values and
    # an array of actions of the same length
    # an array of rewards of the same length
    def train(self, trajectories, learning_rate):

        delta_J = [0]

        layer = 0
        for layer_weight_matrix in self.weights:
            layer_weight_matrix += learning_rate * delta_J[layer]
            layer += 1
