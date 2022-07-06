import argparse
import os.path as osp

from tqdm import tqdm

import torch

from datasets.ply_dataset import gen_ply, get_num_classes
from models.polygen import VertexModel
from utils.config import read_config


def run_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vtx_cfg = cfg["model"]["vertex_model"]

    num_classes = get_num_classes(cfg["data"]["classes_file"])
    print(f"Class conditional model with {num_classes} classes")

    model_path = osp.join(cfg["test"]["vtx_model_file"])
    print(f"Using vertex model: {model_path}")
    model = VertexModel.load_from_checkpoint(model_path)

    model = model.to(device)
    model.eval()

    num_valid = 0

    with torch.no_grad():
        for class_ndx in tqdm(range(num_classes), desc="class"):
            for out_ndx in tqdm(range(cfg["test"]["num_samples"]), desc="sample"):
                vtx_seq = model.greedy_decode(
                    probabilistic=cfg["test"]["probabilistic"],
                    nucleus=cfg["test"]["nucleus"],
                    class_ndx=class_ndx,
                )
                vertices = model.vertexseq_to_raw_vertices(vtx_seq)
                # check if we got valid vertices
                if vertices is not None:
                    num_valid += 1
                    ply_path = osp.join(
                        cfg["data"]["out_path"], f"{class_ndx}_{out_ndx}.ply"
                    )
                    gen_ply(ply_path, vertices)
    print(f"{num_valid} valid meshes generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_inference(cfg)
