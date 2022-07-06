import argparse
import os.path as osp

from tqdm import tqdm

import torch
from torchvision.transforms import Compose

from transforms.polygen import (
    SortMesh,
    MeshToSeq,
    QuantizeVertices8Bit,
    GetFacePositions,
    MeshToTensor,
)
from datasets.ply_dataset import PLYDataset, gen_ply
from datasets.utils import get_dataset
from models.polygen import FaceModel
from utils.config import read_config


def run_inference(cfg):
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaceModel(face_cfg, vtx_cfg)
    model_path = osp.join(cfg["test"]["face_model_file"])
    print(f"Using model: {model_path}")
    state_dict = torch.load(model_path)["state_dict"]
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    ######
    # prepare vertices dataset
    t = Compose([QuantizeVertices8Bit(remove_duplicates=True, scale=32), SortMesh()])

    dataset = PLYDataset(cfg["data"]["data_path"], cfg["data"]["val_list"], transform=t)
    ######

    num_valid = 0

    with torch.no_grad():
        for out_ndx in tqdm(range(cfg["test"]["num_samples"])):
            vertices = dataset[out_ndx]["vertices"]
            face_seq = model.greedy_decode(vertices, probabilistic=True).numpy()
            # check if face seq is valid
            if len(face_seq) > 0:
                faces = model.faceseq_to_raw_faces(face_seq)
                num_valid += 1
                ply_path = osp.join(cfg["data"]["out_path"], f"{out_ndx}.ply")
                gen_ply(ply_path, vertices, faces)
    print(f"{num_valid} valid meshes generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_inference(cfg)
