import argparse
from pathlib import Path
from collections import defaultdict
import os.path as osp

from tqdm import tqdm

##############
# Set seeds
import numpy as np

np.random.seed(0)

import torch

torch.manual_seed(0)
##############

from torchvision.transforms import Compose
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler

from datasets.utils import get_dataset
from datasets.ply_dataset import PLYDataset, plydataset_collate_fn, gen_ply
from transforms.polygen import (
    SortMesh,
    MeshToSeq,
    QuantizeVertices8Bit,
    GetFacePositions,
    MeshToTensor,
)

from utils.config import read_config
from models.polygen import FaceModel
from models.misc import count_parameters, get_lr

MODEL_COMPONENTS = ("face",)
SPLITS = ("train", "val")


def eval_face_model(faces_gt, faces_logprobs, face_cfg):
    loss = 0
    for ndx, preds in enumerate(faces_logprobs):
        gt = faces_gt[ndx].clone()
        # remove start tokens from gt - we dont want to predict these finally
        gt = gt[1:]
        # -1 score is the stop face token
        gt[gt == face_cfg["face_stop_token"]] = preds.shape[1] - 1
        # -2 score is the new face token
        gt[gt == face_cfg["new_face_token"]] = preds.shape[1] - 2

        loss += F.nll_loss(preds, gt, ignore_index=face_cfg["face_pad_token"])
    return loss


def run_training(cfg):
    """
    cfg: full cfg for polygen training
    """
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### prepare data #####
    t = Compose(
        [
            QuantizeVertices8Bit(remove_duplicates=True, scale=32),
            SortMesh(),
            GetFacePositions(new_face_token=face_cfg["new_face_token"]),
            MeshToSeq(
                vtx_start_token=vtx_cfg["vertex_start_token"],
                vtx_stop_token=vtx_cfg["vertex_stop_token"],
                vtx_pad_token=vtx_cfg["vertex_pad_token"],
                vtx_seq_len=vtx_cfg["max_seq_len"],
                new_face_token=face_cfg["new_face_token"],
                face_pad_token=face_cfg["face_pad_token"],
                face_seq_len=face_cfg["max_face_seq_len"],
                face_stop_token=face_cfg["face_stop_token"],
            ),
            MeshToTensor(),
        ]
    )

    train_set = PLYDataset(
        cfg["data"]["data_path"], cfg["data"]["train_list"], transform=t
    )
    # train_set = Subset(train_set, range(128))
    val_set = PLYDataset(cfg["data"]["data_path"], cfg["data"]["val_list"], transform=t)
    # val_set = Subset(val_set, range(128))

    print(f"Train set: {len(train_set)}")
    print(f"Val set: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        collate_fn=plydataset_collate_fn,
        drop_last=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["val_batch_size"],
        collate_fn=plydataset_collate_fn,
        drop_last=True,
        num_workers=4,
    )
    loaders = {"train": train_loader, "val": val_loader}
    ##### done preparing dataset ######
    model = FaceModel(face_cfg, vtx_cfg).to(device)
    print(f"Model parameters: {count_parameters(model)}")

    optim_face = Adam(
        model.parameters(),
        lr=cfg["train"]["face"]["lr"],
        weight_decay=cfg["train"]["face"]["l2"],
    )
    # reset LR every 20 epochs, reduce till lr=0
    sched_face = CosineAnnealingLR(optim_face, T_max=20)
    face_warmup = GradualWarmupScheduler(
        optim_face, multiplier=1, total_epoch=20, after_scheduler=sched_face
    )

    writer = SummaryWriter()

    for epoch in tqdm(range(cfg["train"]["epochs"])):
        # dict['train'/'val']['vtx'/'face'/'total']
        losses = {split: defaultdict(int) for split in loaders}
        accs = {split: defaultdict(int) for split in loaders}

        for split in loaders:
            if split == "train":
                model.train()
            else:
                model.eval()
            for batch_ndx, batch in enumerate(loaders[split]):
                # loss in this batch
                batch_loss = 0
                if split == "train":
                    optim_face.zero_grad()
                # get all the inputs that we need
                vertices_raw, faces_raw = batch["vertices_raw"], batch["faces_raw"]
                vertices, faces = (
                    batch["vertices"].to(device),
                    batch["faces"].to(device),
                )
                face_pos, in_face_pos = (
                    batch["face_pos"].to(device),
                    batch["in_face_pos"].to(device),
                )
                # forward pass
                face_logprobs = model(
                    face_pos, in_face_pos, faces, vertices, vertices_raw
                )
                # evaluate
                face_loss = eval_face_model(
                    faces, face_logprobs, cfg["model"]["face_model"]
                )

                # backprop
                if split == "train":
                    face_loss.backward()
                    clip_grad_norm_(model.parameters(), 1.0)
                    optim_face.step()
                losses[split]["face"] += face_loss.detach().cpu().numpy()
        face_warmup.step()
        writer.add_scalar("LR/face", get_lr(optim_face), epoch)

        for split in losses:
            for component in losses[split]:
                # divide by the size of the dataset
                losses[split][component] /= len(loaders[split].dataset)
        # write everything to tensorboard
        for component in MODEL_COMPONENTS:
            writer.add_scalars(
                f"Loss/{component}",
                {split: losses[split][component] for split in SPLITS},
                epoch,
            )
        if epoch % 1 == 0:
            out_path = osp.join(cfg["data"]["model_dir"], f"output_{epoch}.pth")
            print(f"Save: {out_path}")
            save_model(model, out_path)

    save_model(model, osp.join(cfg["data"]["model_dir"], "output_final.pth"))


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_training(cfg)
