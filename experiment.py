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


if __name__ == "__main__":
    options = {
        "framework.seed": [12345],
        "dm_model.model.backbone": ["dlstm"],
        "model.backbone": ["fc", "dfc", "gru", "dgru", "lstm", "dlstm"],
        "train.LAMBDA": [0.0, 0.01, 0.1, 0.15],
    }
    config, num_workers = get_configs(False)
    main(config, num_workers, options)
