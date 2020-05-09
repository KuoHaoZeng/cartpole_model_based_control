from utils.get_all_experiments_results import (
    output_jsonlines_results,
    output_txt_results,
)
from scripts.experiments_helper import (
    Worker,
    get_all_possible_experiments_options,
    get_configs,
)
from multiprocessing import Queue
from multiprocessing import set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass


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
    options = {
        "dm_model.model.backbone": ["dfc", "dlstm", "dgru"],
        "model.backbone": ["fc", "dfc", "gru", "dgru", "lstm", "dlstm"],
        "train.LAMBDA": [0.0, 0.01, 0.1, 0.15],
    }
    config, num_workers = get_configs()
    main(config, num_workers, options)
