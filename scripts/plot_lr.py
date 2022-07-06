"""
Script to visualize different learning rate schedulers
Add your scheduler below and look at the LR values in tensorboard
"""

import argparse
from pathlib import Path
from collections import defaultdict
import os.path as osp

from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import (
    MultiplicativeLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler

from utils.config import read_config
from models.polygen import PolyGen

from models.misc import get_lr


def run_training(cfg):
    """
    cfg: full cfg for polygen training
    """
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolyGen(vtx_cfg, face_cfg).to(device)

    optim_vertex = Adam(
        model.vertex_model.parameters(),
        lr=cfg["train"]["vtx"]["lr"],
        weight_decay=cfg["train"]["vtx"]["l2"],
    )

    optim_face = Adam(
        model.face_model.parameters(),
        lr=cfg["train"]["face"]["lr"],
        weight_decay=cfg["train"]["face"]["l2"],
    )

    # sched_vtx = CosineAnnealingLR(optim_vertex, T_max=25)
    sched_vtx = CosineAnnealingWarmRestarts(optim_vertex, T_0=25, eta_min=1e-5)
    vtx_warmup = GradualWarmupScheduler(
        optim_vertex, multiplier=1, total_epoch=20, after_scheduler=sched_vtx
    )
    sched_face = CosineAnnealingLR(optim_face, T_max=20)

    writer = SummaryWriter(max_queue=1)

    for epoch in tqdm(range(cfg["train"]["epochs"])):
        optim_vertex.zero_grad()
        optim_face.zero_grad()

        optim_vertex.step()
        optim_face.step()

        # sched_vtx.step()
        vtx_warmup.step()
        sched_face.step()

        writer.add_scalar(f"LR/vertex", get_lr(optim_vertex), epoch)
        writer.add_scalar(f"LR/face", get_lr(optim_face), epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_training(cfg)
