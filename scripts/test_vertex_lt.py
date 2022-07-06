import argparse
import os.path as osp

from tqdm import tqdm

import torch

from datasets.ply_dataset import gen_ply
from models.polygen import VertexModel
from utils.config import read_config


def run_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vtx_cfg = cfg["model"]["vertex_model"]

    model_path = osp.join(cfg["test"]["vtx_model_file"])
    state_dict = torch.load(model_path)["state_dict"]
    model = VertexModel(vtx_cfg)
    model.load_state_dict(state_dict)

    print(f"Using model: {model_path}")

    model = model.to(device)
    model.eval()

    num_valid = 0

    with torch.no_grad():
        for out_ndx in tqdm(range(cfg["test"]["num_samples"])):
            vtx_seq = model.greedy_decode(probabilistic=cfg["test"]["probabilistic"])
            vertices = model.vertexseq_to_raw_vertices(vtx_seq)
            # check if we got valid vertices
            if vertices is not None:
                num_valid += 1
                ply_path = osp.join(cfg["data"]["out_path"], f"{out_ndx}.ply")
                gen_ply(ply_path, vertices)
    print(f"{num_valid} valid meshes generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_inference(cfg)
