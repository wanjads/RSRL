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
        # print("inp: " + str(inp))
        opt = self.model.predict(inp)
        # print("before: " + str(opt))
        target = cost + constants.gamma * opt[0][action]
        opt[0][action] = target
        # print("opt: " + str(opt))
        history = self.model.fit(inp, opt, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        # print("after: " + str(self.model.predict(inp)))
        return loss
