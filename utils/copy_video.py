import os
import sys
import subprocess


def get_all_mp4(base_dir):
    return [
        (ele1, ele2, ele3)
        for ele1, ele2, ele3 in os.walk(base_dir)
        if len(ele3) > 0 and ele3[0].endswith(".mp4")
    ]


def copy_to_target_dir(videos, target_dir):
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    for mp4 in videos:
        path = "{}/{}".format(mp4[0], mp4[2][0])
        target_path = "{}/{}".format(target_dir, mp4[2][0])
        cmd = "cp {} {}".format(path, target_path)
        print(cmd)
        _ = subprocess.call("cp {} {}".format(path, target_path), shell=True)


if __name__ == "__main__":
    base_dir, target_dir = sys.argv[1], sys.argv[2]
    videos = get_all_mp4(base_dir)
    copy_to_target_dir(videos, target_dir)
