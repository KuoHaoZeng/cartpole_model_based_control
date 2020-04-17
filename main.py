import argparse
import torch
from utils.config import Config

class Trainer():
    def __init__(self, configs):

        ### somethings
        self.cfg = configs

        ### logging

        ### define loss functions

        ### model


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
    trainer = Trainer(cfg)

if __name__ == '__main__':
    config = get_configs()
    main(config)
