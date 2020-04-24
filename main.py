import argparse
import torch
from utils.config import Config

from network import network
from data import dataset

class Trainer():
    def __init__(self, configs):

        ### somethings
        self.cfg = configs

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

class Tester():
    def __init__(self, configs):

        ### somethings
        self.cfg = configs

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

def get_configs():
    parser = argparse.ArgumentParser(description="We are the best!!!")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    config = Config(args.config)
    return config

def main(cfg):
    state_dataset = dataset.state_dataset(cfg)
    img_dataset = dataset.image_dataset(cfg)
    trainer = Trainer(cfg)

    #for idx, (s, x, y) in enumerate(state_dataset):
    #    print("index: {}".format(idx))
    #    print("current augmented state:")
    #    print(x.size())
    #    print("ground truth delta state:")
    #    print(y.size())

    #for idx, (imgs, s, x, y) in enumerate(img_dataset):
    #    print("index: {}".format(idx))

if __name__ == '__main__':
    config = get_configs()
    main(config)
