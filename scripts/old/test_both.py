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
from datasets.ply_dataset import gen_ply
from datasets.utils import get_dataset
from models.polygen import VertexModel, FaceModel
from utils.config import read_config


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def run_inference(cfg):
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vtx_model = VertexModel(vtx_cfg)
    vtx_model_path = osp.join(cfg["test"]["vtx_model_file"])
    print(f"Using vertex model: {vtx_model_path}")
    vtx_model = load_model(vtx_model, vtx_model_path)

    face_model = FaceModel(face_cfg, vtx_cfg)
    face_model_path = osp.join(cfg["test"]["face_model_file"])
    print(f"Using face model: {face_model_path}")
    face_model = load_model(face_model, face_model_path)

    vtx_model, face_model = vtx_model.to(device), face_model.to(device)
    vtx_model.eval()
    face_model.eval()

    num_valid = 0

    with torch.no_grad():
        for out_ndx in tqdm(range(cfg["test"]["num_samples"])):
            vtx_seq = vtx_model.greedy_decode(
                probabilistic=cfg["test"]["probabilistic"]
            )
            vertices = vtx_model.vertexseq_to_raw_vertices(vtx_seq)
            # check if we got valid vertices
            if vertices is not None:
                face_seq = face_model.greedy_decode(
                    vertices, probabilistic=cfg["test"]["probabilistic"]
                ).numpy()
                # check if face seq is valid
                if len(face_seq) > 0:
                    faces = face_model.faceseq_to_raw_faces(face_seq)
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
