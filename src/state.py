import random
import constants


# a state of the MDP
class State:

    # a state consists of the age of information at the sender and at the receiver
    # + the information, if the sender tried to send a package in the last episode
    def __init__(self, aoi_sender, aoi_receiver, last_action):
        self.aoi_sender = aoi_sender
        self.aoi_receiver = aoi_receiver
        self.last_action = last_action

    # a state update dependent on the decision to send
    def update(self, action):
        if self.aoi_sender < constants.aoi_cap:
            self.aoi_sender += 1
        if self.aoi_receiver < constants.aoi_cap:
            self.aoi_receiver += 1
        if random.random() < constants.new_package_prob:
            self.aoi_sender = 0
        self.last_action = 0
        if action:
            self.last_action = 1
            if random.random() < constants.send_prob and self.aoi_sender < constants.aoi_cap:
                self.aoi_receiver = self.aoi_sender + 1

        return self

    # the initial state
    @staticmethod
    def initial_state():
        return State(0, 1, 0)
