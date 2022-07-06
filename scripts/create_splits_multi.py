import argparse
import os, os.path as osp
import numpy as np
from pathlib import Path
from datasets.ply_dataset import PLYDataset


class PLYPath:
    def __init__(self, path=None, vals=None):
        if path:
            self.root, self.objid, self.augndx = [path.parent] + path.stem.split("_")
        elif vals:
            self.root, self.objid, self.augndx = vals

    @property
    def path(self):
        return osp.join(self.root, f"{self.objid}_{self.augndx}.ply")


def get_val_path_objs(path_objs, val_ids):
    """
    path_objs: list of PLYPath objects
    val_ids: allowed obj ids

    return: list of PLYPaths with objid in val_ids and augndx=0
    """
    root_objid_tups = set([(p.root, p.objid) for p in path_objs if p.objid in val_ids])

    return [PLYPath(vals=(t[0], t[1], 0)) for t in root_objid_tups]


def main(args):
    # create a dataset to get the filepaths relative to the root
    dataset = PLYDataset(args.data_path, multi_objs=True)
    # get all PLY filenames
    files = list(map(Path, dataset.files))

    ## these are of the form: 04256520/d4aabbe3527c84f670793cd603073927_2.ply
    ## path.parent: 04256520 is the classID
    ## path.stem: d4aabbe3527c84f670793cd603073927_2.ply

    # get the key (root, x, y) from the filename 'root/x_y.ply'
    path_objs = list(map(PLYPath, files))
    # unique mesh ids (only x)
    mesh_ids = list(set([p.objid for p in path_objs]))
    print(f"Num meshes: {len(mesh_ids)}")
    # number of train meshes
    num_train_meshes = int(args.train_frac * len(mesh_ids))

    # objid of train meshes
    train_mesh_ids = mesh_ids[:num_train_meshes]
    # objids of val meshes
    val_mesh_ids = mesh_ids[num_train_meshes:]

    # path objs of train meshes
    train_path_objs = list(filter(lambda p: p.objid in train_mesh_ids, path_objs))
    # (root, objid, 0) for val meshes
    # val set is not augmented: select the first mesh in the augmented set
    val_path_objs = get_val_path_objs(path_objs, val_mesh_ids)

    train_len = len(train_path_objs)
    val_len = len(val_path_objs)
    total_len = train_len + val_len

    print(f"Total: {total_len}, train: {train_len}, val: {val_len}")

    splits = {"train": train_path_objs, "val": val_path_objs}
    for split in splits:
        with open(f"{split}.txt", "w") as f:
            print(f"Writing to {f.name}")
            f.writelines((p.path + "\n" for p in splits[split]))
    print("Files created in current directory, don't forget to move them!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train val splits for dataset with multiple object classes"
    )
    parser.add_argument("data_path", help="Data folder")
    parser.add_argument("train_frac", type=float, help="Training fraction")
    args = parser.parse_args()

    main(args)
