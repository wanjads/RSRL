import copy
import constants
import numpy as np


class NN:

    def __init__(self, input_size):
        self.W1 = 0.1 * np.random.rand(10, input_size + 1)
        self.W2 = 0.1 * np.random.rand(2, 10 + 1)
        self.learning_rate = 0.01

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_der(x):
        return np.greater(x, 0)*1

    def out(self, inp):

        inp = np.concatenate((inp, [1]))

        zh = np.dot(self.W1, inp)
        ah = self.relu(zh)

        ah = np.concatenate((ah, [1]))

        out = np.dot(self.W2, ah)

        return out

    def train_model(self, inp, action, cost):

        cost = (cost - 4) / 4

        # feedforward

        inp = np.concatenate((inp, [1]))

        zh = np.dot(self.W1, inp)
        ah = self.relu(zh)
        ah = np.concatenate((ah, [1]))

        out = np.dot(self.W2, ah)
        target_vector = copy.deepcopy(out)

        target = cost + constants.gamma * out[action]
        print(target)
        target_vector[action] = target

        # Phase1 =======================
        error_out = ((1 / 2) * (np.power((out - target_vector), 2)))
        # print(error_out)

        dcost_dout = out - target_vector
        dout_dW2 = ah

        dcost_dW2 = np.dot(dcost_dout.reshape(2, 1), dout_dW2.reshape(1, 11))

        # Phase 2 =======================

        dout_dah = self.W2
        dcost_dah = np.dot(dcost_dout.reshape(1, 2), dout_dah.reshape(2, 11))
        dcost_dah = dcost_dah.reshape(11,)[:10]
        dah_dzh = np.diag(self.relu_der(zh)[:10])
        dcost_dzh = np.dot(dcost_dah.reshape(1, 10), dah_dzh.reshape(10, 10))
        dzh_dW1 = inp
        dcost_dW1 = np.dot(dcost_dzh.reshape(10, 1), dzh_dW1.reshape(1, 4))

        # Update Weights ================

        self.W1 += self.learning_rate * dcost_dW1
        self.W2 += self.learning_rate * dcost_dW2
