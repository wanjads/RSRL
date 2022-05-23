# imports
import random
import constants


# a state of the MDP
class State:

    # a state consists of the age of information at the sender and at the receiver
    # and two estimates for the probabilities that a new package arrives resp. that sending is successful
    def __init__(self, aoi_sender, aoi_receiver, new_pack_prob_estimate, send_prob_estimate):
        self.aoi_sender = aoi_sender
        self.aoi_receiver = aoi_receiver
        self.new_pack_prob_estimate = new_pack_prob_estimate
        self.send_prob_estimate = send_prob_estimate

    # a state update dependent on the decision to send
    def update(self, action):
        self.aoi_sender += 1
        self.aoi_receiver += 1
        if random.random() < constants.new_package_prob:
            self.aoi_sender = 0
        if action:
            if random.random() < constants.send_prob:
                self.aoi_receiver = self.aoi_sender + 1

        return self

    # the initial state
    @staticmethod
    def initial_state():
        return State(0, 0, 0.5, 0.5)
