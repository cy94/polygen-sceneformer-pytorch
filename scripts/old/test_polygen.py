import argparse
import os.path as osp

from tqdm import tqdm

import torch

from datasets.ply_dataset import gen_ply
from models.polygen import PolyGen
from utils.config import read_config


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def run_inference(cfg):
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolyGen(vtx_cfg, face_cfg)
    model_path = osp.join(cfg["data"]["model_dir"], cfg["test"]["model_file"])
    model = load_model(model, model_path)

    print(f"Using model: {model_path}")

    model = model.to(device)
    model.eval()

    num_valid = 0

    with torch.no_grad():
        for out_ndx in tqdm(range(cfg["test"]["num_samples"])):
            vertices, faces = model.greedy_decode(
                probabilistic=cfg["test"]["probabilistic"]
            )
            # check if we got valid vertices and faces
            if vertices is not None and faces is not None:
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
