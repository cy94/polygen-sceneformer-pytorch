import argparse
import os
import numpy as np
from pathlib import Path
from utils.config import read_config


def xy_to_filename(xy):
    return f"{xy[0]}_{xy[1]}.ply"


def main(cfg, train_frac):
    path = Path(cfg["data"]["data_path"])
    # get all PLY filenames
    files = [f for f in os.listdir(path) if f.endswith(".ply")]
    # sort 'x_y.ply' by the key (x, y)
    getkey = lambda f: list(map(int, Path(f).stem.split("_")))
    # get (x, y) tuples
    file_keys = sorted(map(getkey, files))
    # unique mesh ids (only x)
    mesh_ids = list(set([key[0] for key in file_keys]))
    print(f"Num meshes: {len(mesh_ids)}")
    # number of train meshes
    num_train_meshes = int(train_frac * len(mesh_ids))

    # (x) of train meshes
    train_mesh_ids = mesh_ids[:num_train_meshes]
    # (x) of val meshes
    val_mesh_ids = mesh_ids[num_train_meshes:]
    # (x, y) of train meshes
    train_set_keys = filter(lambda k: k[0] in train_mesh_ids, file_keys)
    # (x, 0) tuples of of val meshes
    # val set is not augmented: select the first mesh in the augmented set
    val_set_keys = [(x, 0) for x in val_mesh_ids]

    train_set = list(map(xy_to_filename, train_set_keys))
    val_set = list(map(xy_to_filename, val_set_keys))

    train_len = len(train_set)
    total_len = len(train_set) + len(val_set)

    print(f"Total: {total_len}, train: {train_len}, val: {total_len-train_len}")

    splits = {"train": train_set, "val": val_set}
    for split in splits:
        with open(f"{split}.txt", "w") as f:
            print(f"Writing to {f.name}")
            f.writelines((l + "\n" for l in splits[split]))
    print("Files created in current directory, don't forget to move them!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    parser.add_argument("train_frac", type=float, help="Training fraction")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    main(cfg, args.train_frac)
