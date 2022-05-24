from state import State


class Strategy:

    def __init__(self):
        self.new_package_prob_estimate = 0
        self.sending_prob_estimate = 0
        self.no_of_tries_to_send = 0
        # here, we store the weights of the network

    def update(self, state, action, episode_no):

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

    def action(self, state, episode_no):
        return 1
