import argparse
import os
from pathlib import Path
from collections import defaultdict
import os.path as osp
from itertools import chain
from tqdm import tqdm
from tqdm.auto import trange

import torch
import numpy as np
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Subset

from datasets.utils import get_dataset
from datasets.ply_dataset import PLYDataset, plydataset_collate_fn, gen_ply
from transforms.polygen import (
    SortMesh,
    MeshToSeq,
    QuantizeVertices8Bit,
    GetFacePositions,
    MeshToTensor,
    FilterMesh,
    RandomScale,
    RandomLinearWarp,
    YUpToZUp,
)
from utils.config import read_config


def augment(args):
    """
    args: full args from command line
    """
    ##### prepare data #####
    t = Compose([YUpToZUp(), RandomScale(), RandomLinearWarp(warp_var=0.1)])

    dataset = PLYDataset(args.input_path)

    os.makedirs(args.output_path, exist_ok=True)

    # each mesh
    for mesh in tqdm(dataset, desc="mesh"):
        stem = Path(mesh["filename"]).stem
        # create 50 augmented versions
        for augndx in range(50):
            augmesh = t(mesh)
            vtx, faces = augmesh["vertices"], augmesh["faces"]
            ply_path = osp.join(args.output_path, f"{stem}_{augndx}.ply")
            gen_ply(ply_path, vtx, faces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read PLY from file, augment and write back to file"
    )
    parser.add_argument("input_path", help="Path to input PLY files")
    parser.add_argument("output_path", help="Path to input PLY files")

    args = parser.parse_args()

    augment(args)
