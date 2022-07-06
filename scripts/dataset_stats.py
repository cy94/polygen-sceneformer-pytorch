import argparse
from pathlib import Path
from collections import defaultdict
import os.path as osp
from itertools import chain
from tqdm import tqdm

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
)
from utils.config import read_config


def get_stats(args):
    """
    args: args from command line
    """
    ##### prepare data #####
    t = Compose([QuantizeVertices8Bit(remove_duplicates=True, scale=255),])

    print(f"Multiple objects in data folder: {args.multi_objs}")
    dataset = PLYDataset(
        args.data_folder,
        transform=t,
        multi_objs=args.multi_objs,
        classes_file=args.classes_file,
    )

    print(f"Dataset size: {len(dataset)}")

    if not args.viz:
        print("Finding stats .. ")
        vtx_per_mesh = []
        faces_per_mesh = []
        vtx_per_face = []

        for ndx in tqdm(range(len(dataset))):
            # filter out bad meshes here
            mesh = dataset[ndx]

            vertices, faces = mesh["vertices"], mesh["faces"]
            vtx_per_mesh.append(len(vertices))
            faces_per_mesh.append(len(faces))
            vtx_per_face.append([len(face) for face in faces])
        print("done!")

        stats = {
            "vtx_per_mesh": vtx_per_mesh,
            "face_per_mesh": faces_per_mesh,
            "vtx_per_face": vtx_per_face,
        }
        out_file = "stats.pth"
        print(f"Saving to: {out_file}")
        torch.save(stats, out_file)

    if args.viz:
        print("Visualize only")
        print("Writing meshes to file ..")
        # write the transformed meshes to PLY files
        for ndx, mesh in enumerate(dataset):
            vtx, faces = mesh["vertices"], mesh["faces"]
            ply_path = osp.join(args.out_path, f"{ndx}.ply")
            gen_ply(ply_path, vtx, faces)
            print(ndx)
            # only the first 10
            if ndx == 100:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help="Root folder of dataset")
    parser.add_argument(
        "--multi_objs",
        action="store_true",
        help="Data folder contains multiple object classes",
    )
    parser.add_argument("--classes_file", help="Path to classes file", default=None)
    parser.add_argument(
        "--viz", action="store_true", help="Store transformed meshes to file?"
    )
    parser.add_argument("--out_path", help="Path to transformed meshes")

    args = parser.parse_args()

    get_stats(args)
