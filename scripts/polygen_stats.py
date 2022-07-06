"""
Compute the statistics in Figure 7 of the PolyGen paper
https://arxiv.org/pdf/2002.10880.pdf
"""

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

import pymesh
import torch_geometric as ptgeom


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

    print("Finding stats .. ")
    num_vertices, num_faces, num_comps = [], [], []
    degrees, edge_lengths, face_areas = [], [], []

    for mesh_path in tqdm(dataset.files):
        # filter out bad meshes here
        full_path = osp.join(dataset.root, mesh_path)
        mesh = pymesh.load_mesh(full_path)

        mesh.enable_connectivity()
        # num_vertices.append(mesh.num_vertices)
        # num_faces.append(mesh.num_faces)
        # num_comps.append(mesh.num_components)
        # print(mesh.num_vertices, mesh.num_faces) #, mesh.num_components, len(mesh.face_area))
        # mesh.enable_connectivity()
        # g_vertices, g_edges = pymesh.mesh_to_graph(mesh)
        # graph = ptgeom.Data(pos=g_vertices, edge_index=g_edges)

    print("done!")

    stats = {
        "vertices": num_vertices,
        "faces": num_faces,
        "components": num_comps,
        "degree": degrees,
        "edge_length": edge_lengths,
        "face_area": face_areas,
    }
    out_file = "stats.pth"
    print(f"Saving to: {out_file}")
    torch.save(stats, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help="Root folder of dataset")
    parser.add_argument(
        "--multi_objs",
        action="store_true",
        help="Data folder contains multiple object classes",
    )
    parser.add_argument("--classes_file", help="Path to classes file", default=None)

    args = parser.parse_args()

    get_stats(args)
