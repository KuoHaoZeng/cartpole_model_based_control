from network import network
from data import dataset

model_protocol = {"state": network.model_state, "image": network.model_CNN}
dataset_protocol = {"state": dataset.state_dataset, "image": dataset.image_dataset}


class Tester:
    def __init__(self, config):

        ### somethings
        self.cfg = config

        ### logging

        ### define loss functions

        ### model
        self.model = network.model_CNN(config)

    def save_checkpoints(self):
        raise NotImplementedError

    def load_checkpoints(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
