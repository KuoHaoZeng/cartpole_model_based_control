import argparse
from utils.config import Config
from utils.get_all_experiments_results import output_jsonlines_results, output_txt_results
from utils.experiments_helper import Worker, get_all_possible_experiments_options
from multiprocessing import Queue


def get_configs():
    parser = argparse.ArgumentParser(description="We are the best!!!")
    parser.add_argument("--config", type=str, default="configs/dm_state.yaml")
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()
    config = Config(args.config, False)
    return config, args.n


def main(config, num_workers, options):
    queue = Queue()
    queue = get_all_possible_experiments_options(options, queue)

    workers = []
    for n in range(num_workers):
        workers.append(Worker(n, config, queue))

    for n in range(num_workers):
        workers[n].start()

    for n in range(num_workers):
        workers[n].join()

    output_jsonlines_results("results/{}".format(config.exp_prefix))
    output_txt_results("results/{}".format(config.exp_prefix))


if __name__ == "__main__":
    options = {"dm_model.model.backbone": ["dfc", "dlstm", "dgru"],
               "model.backbone": ["fc", "dfc", "gru", "dgru", "lstm", "dlstm"],
               "train.LAMBDA": [0.0, 0.01, 0.1]}
    config, num_workers = get_configs()
    main(config, num_workers, options)
