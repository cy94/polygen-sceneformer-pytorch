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
from datasets.ply_dataset import gen_ply, get_num_classes
from datasets.utils import get_dataset
from models.polygen import VertexModel, FaceModel
from utils.config import read_config


def load_model(model, path):
    model.load_state_dict(torch.load(path)["state_dict"])
    return model


def run_inference(cfg):
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = get_num_classes(cfg["data"]["classes_file"])
    print(f"Class conditional model with {num_classes} classes")

    vtx_path = cfg["test"]["vtx_model_file"]
    print(f"Using vertex model: {vtx_path}")
    vtx_model = VertexModel.load_from_checkpoint(vtx_path)

    face_path = cfg["test"]["face_model_file"]
    print(f"Using face model: {face_path}")
    face_model = FaceModel.load_from_checkpoint(face_path)

    vtx_model, face_model = vtx_model.to(device), face_model.to(device)
    vtx_model.eval()
    face_model.eval()

    num_valid = 0

    with torch.no_grad():
        for class_ndx in tqdm(range(num_classes), desc="class"):
            num_valid_class = 0
            for out_ndx in tqdm(range(cfg["test"]["num_samples"])):
                vtx_seq = vtx_model.greedy_decode(
                    probabilistic=cfg["test"]["probabilistic"],
                    nucleus=cfg["test"]["nucleus"],
                    class_ndx=class_ndx,
                )
                vertices = vtx_model.vertexseq_to_raw_vertices(vtx_seq)
                # check if we got valid vertices
                if vertices is not None:
                    face_seq = face_model.greedy_decode(
                        vertices,
                        probabilistic=cfg["test"]["probabilistic"],
                        nucleus=cfg["test"]["nucleus"],
                        class_ndx=class_ndx,
                    ).numpy()
                    # check if face seq is valid
                    if len(face_seq) > 0:
                        faces = face_model.faceseq_to_raw_faces(face_seq)
                        num_valid += 1
                        num_valid_class += 1
                        ply_path = osp.join(
                            cfg["data"]["out_path"], f"{class_ndx}_{out_ndx}.ply"
                        )
                        gen_ply(ply_path, vertices, faces)
            print(f"Class: {class_ndx}, valid meshes: {num_valid_class}")
    print(f"{num_valid} valid meshes generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_inference(cfg)
