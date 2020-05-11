import json, os, sys
import numpy as np


def get_all_result_files(base_dir):
    files = ["{}/{}".format(base_dir, ele) for ele in os.listdir(base_dir)]
    possbile_files = []
    for f in files:
        if os.path.isdir(f):
            possbile_files += get_all_result_files(f)
        elif f.endswith(".json"):
            possbile_files.append(f)
    return sorted(possbile_files)


def output_jsonlines_results(base_dir):
    possible_files = get_all_result_files(base_dir)
    f = open("{}/experiment_results.jsonl".format(base_dir), "wb")
    for result in possible_files:
        exp_option = result.split("/")[-2]
        rollout_loss = json.load(open(result, "r"))["rollout_L2_Loss"]
        output = json.dumps(
            {exp_option: {"mean": np.mean(rollout_loss), "std": np.std(rollout_loss)}},
            ensure_ascii=False,
        )
        f.write(output.encode("utf-8"))
        f.write("\n".encode("utf-8"))


def output_txt_results(base_dir, std=True):
    possible_files = get_all_result_files(base_dir)
    f = open("{}/experiment_results.txt".format(base_dir), "w")
    results = {}
    for result in possible_files:
        exp_option = result.split("/")[-2]
        rollout_loss = json.load(open(result, "r"))["rollout_L2_Loss"]
        LAMBDA = exp_option.split("_")[3].split("lambda")[0]
        dm_model = exp_option.split("_")[1]
        policy_network = exp_option.split("_")[0]
        if LAMBDA not in results.keys():
            results[LAMBDA] = {}
        if dm_model not in results[LAMBDA].keys():
            results[LAMBDA][dm_model] = {}
        if std:
            results[LAMBDA][dm_model][policy_network] = "{:.3f}±{:.3f}".format(
                np.mean(rollout_loss), np.std(rollout_loss)
            )
        else:
            results[LAMBDA][dm_model][policy_network] = "{:.3f}".format(
                np.mean(rollout_loss)
            )

    for LAMBDA, v in results.items():
        f.write("{}\n".format(LAMBDA))
        for i, (dm_model, sub_v) in enumerate(v.items()):
            if i == 0:
                f.write("\t")
                for policy_network in sub_v.keys():
                    f.write("{}\t".format(policy_network))
                f.write("\n")
            f.write("{}\t".format(dm_model))
            for _, result in sub_v.items():
                f.write("{}\t".format(result))
            f.write("\n")
        f.write("\n")


if __name__ == "__main__":
    base_dir = sys.argv[1]
    output_jsonlines_results(base_dir)
    output_txt_results(base_dir, False)
