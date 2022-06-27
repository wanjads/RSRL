import numpy as np
from keras.models import Input, Model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import constants


class NN:

    def __init__(self, input_size):
        self.inputs = Input(shape=(input_size,))
        self.x = Dense(10, activation='relu')(self.inputs)
        self.output = Dense(2, activation='softmax')(self.x)
        self.model = Model(self.inputs, self.output)
        opt = Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss='mean_squared_error')

    def out(self, inp):
        return self.model.predict(inp)

    def train_model(self, inp, action, cost):
        opt = self.model.predict(inp)
        target = cost + constants.gamma * opt[0][action]
        opt[0][action] = target
        history = self.model.fit(inp, opt, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        return loss


class ReinforceNN:

    def __init__(self, input_size):
        self.parameters = self.init_parameters([input_size, 10, 2])

    def out(self, inp):

        output = np.array(inp)

        for layer_no in range(len(self.parameters)):
            output = np.concatenate((output, np.array([1])))
            output = self.parameters[layer_no] @ output
            if layer_no == len(self.parameters) - 1:
                output = np.exp(output)/np.sum(np.exp(output))
            else:
                output = np.maximum(output, 0)

        return output

    @staticmethod
    def init_parameters(layers):

        parameters = []

        for layer_no in range(len(layers) - 1):
            layer_weight_matrix = np.random.rand(layers[layer_no + 1], layers[layer_no] + 1) - 0.5
            parameters += [layer_weight_matrix]

        return parameters
