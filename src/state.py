import random
import constants


# a state of the MDP
class State:

    # a state consists of the age of information at the sender and at the receiver
    def __init__(self, aoi_sender, aoi_receiver):
        self.aoi_sender = aoi_sender
        self.aoi_receiver = aoi_receiver

    # a state update dependent on the decision to send
    def update(self, action):
        if self.aoi_sender < constants.aoi_cap:
            self.aoi_sender += 1
        if self.aoi_receiver < constants.aoi_cap:
            self.aoi_receiver += 1
        if random.random() < constants.new_package_prob:
            self.aoi_sender = 0
        if action:
            if random.random() < constants.send_prob and self.aoi_sender < constants.aoi_cap:
                self.aoi_receiver = self.aoi_sender + 1

        return self

    # the initial state
    @staticmethod
    def initial_state():
        return State(0, 1)
