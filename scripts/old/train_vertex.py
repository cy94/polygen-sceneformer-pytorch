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
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
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
    FilterMesh,
    RandomScale,
    RandomLinearWarp,
)

from utils.config import read_config
from models.polygen import PolyGen, VertexModel
from models.misc import count_parameters, get_lr

MODEL_COMPONENTS = ("vertex",)
SPLITS = ("train", "val")


def eval_vtx_model(vtx_gt, vtx_logprobs, vtx_cfg):
    """
    TODO: eval over the whole GT and preds together, dont loop
    """
    loss = 0

    for ndx, preds in enumerate(vtx_logprobs):
        gt = vtx_gt[ndx].clone()
        # change the stop token into its class index (subtract 1)
        gt[gt == vtx_cfg["vertex_stop_token"]] -= 1
        # remove start tokens from gt - we dont want to predict these finally
        gt = gt[1:]
        # TODO: ignore padding
        loss += F.nll_loss(preds, gt)

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
    val_set = PLYDataset(cfg["data"]["data_path"], cfg["data"]["val_list"], transform=t)

    # train_set = Subset(train_set, range(128))
    # val_set = Subset(val_set, range(128))

    print(f"Train set: {len(train_set)}")
    print(f"Val set: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        collate_fn=plydataset_collate_fn,
        num_workers=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["val_batch_size"],
        collate_fn=plydataset_collate_fn,
        num_workers=4,
        shuffle=True,
    )
    loaders = {"train": train_loader, "val": val_loader}
    ##### done preparing dataset ######
    model = VertexModel(vtx_cfg).to(device)
    print(f"Model parameters: {count_parameters(model)}")

    optim_vertex = Adam(
        model.parameters(),
        lr=cfg["train"]["vtx"]["lr"],
        weight_decay=cfg["train"]["vtx"]["l2"],
    )
    sched_vtx = CosineAnnealingLR(optim_vertex, T_max=cfg["train"]["epochs"])
    # sched_vtx = CosineAnnealingWarmRestarts(optim_vertex, T_0=cfg['train']['epochs'],
    # eta_min=0)
    vtx_warmup = GradualWarmupScheduler(
        optim_vertex, multiplier=1, total_epoch=20, after_scheduler=sched_vtx
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
                    optim_vertex.zero_grad()
                # get all the inputs that we need
                vertices = batch["vertices"].to(device)
                # forward pass
                vtx_logprobs = model(vertices)
                # evaluate
                vtx_loss = eval_vtx_model(
                    vertices, vtx_logprobs, cfg["model"]["vertex_model"]
                )

                # backprop
                if split == "train":
                    vtx_loss.backward()
                    clip_grad_norm_(model.parameters(), 1.0)
                    optim_vertex.step()
                losses[split]["vertex"] += vtx_loss.detach().cpu().numpy()
        vtx_warmup.step()
        writer.add_scalar("LR/vertex", get_lr(optim_vertex), epoch)

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
