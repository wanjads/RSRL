import random

from state import State
from network import NN
import numpy as np


class Strategy:

    def __init__(self):
        self.input_size = 5
        self.new_package_prob_estimate = 0.5
        self.sending_prob_estimate = 0.5
        self.no_of_tries_to_send = 0
        # here, we initialize the NN
        self.nn = NN(self.input_size)
        self.inp = []

    def update(self, state, action, cost, episode_no):

        # train nn
        loss = self.nn.train(self.inp, action, cost)

        # did a new package arrive at the sender?
        new_package_sender = (state.aoi_sender == 0)

        # update new package prob estimate
        if new_package_sender:
            new_packages_arrived = (episode_no - 1) * self.new_package_prob_estimate + 1
        else:
            new_packages_arrived = (episode_no - 1) * self.new_package_prob_estimate
        self.new_package_prob_estimate = new_packages_arrived / episode_no

        # was a package successfully sent?
        new_package_receiver = (state.aoi_receiver == state.aoi_sender + 1)
        if action:
            if new_package_receiver:
                successful_sends = self.no_of_tries_to_send * self.sending_prob_estimate + 1
            else:
                successful_sends = self.no_of_tries_to_send * self.sending_prob_estimate
            self.no_of_tries_to_send += 1
            self.sending_prob_estimate = successful_sends / self.no_of_tries_to_send

        return loss

    def state_to_input(self, state):
        return np.array([state.aoi_sender,
                         state.aoi_receiver,
                         state.last_action,
                         self.new_package_prob_estimate,
                         self.sending_prob_estimate]).reshape((1, 5))

    def action(self, state, epsilon):
        self.inp = self.state_to_input(state)
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmin(self.nn.out(self.inp)[0])
        return action
