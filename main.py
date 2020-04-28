import argparse
from utils.config import Config
from scripts import trainer, tester


def get_configs():
    parser = argparse.ArgumentParser(description="We are the best!!!")
    parser.add_argument("--config", type=str, default="configs/dm_train.yaml")
    args = parser.parse_args()
    config = Config(args.config)
    return config


def main(cfg):
    # worker = trainer_protocol[cfg.exp_prefix](cfg)
    # worker.run()
    worker = tester_protocol[cfg.exp_prefix](cfg)
    worker.run()


if __name__ == "__main__":
    trainer_protocol = {
        "policy": trainer.Trainer_policy,
        "dm": trainer.Trainer_dynamic_model,
    }
    tester_protocol = {
        "policy": tester.Tester_policy,
        "dm": tester.Tester_dynamic_model,
    }
    config = get_configs()
    main(config)
