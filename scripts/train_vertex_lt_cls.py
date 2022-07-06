import argparse
from pathlib import Path
from collections import defaultdict
import os.path as osp

from tqdm import tqdm

import numpy as np

np.random.seed(0)

import torch

torch.manual_seed(0)

from torchvision.transforms import Compose
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler

from datasets.utils import get_dataset
from datasets.ply_dataset import (
    PLYDataset,
    plydataset_collate_fn,
    gen_ply,
    get_num_classes,
)
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

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

seed_everything(1)


def log_metrics(self, metrics, step=None):
    for k, v in metrics.items():
        if isinstance(v, dict):
            self.experiment.add_scalars(k, v, step)
        else:
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.experiment.add_scalar(k, v, step)


def monkeypatch_tensorboardlogger(logger):
    import types

    logger.log_metrics = types.MethodType(log_metrics, logger)


def run_training(cfg, subset, dev):
    """
    cfg: full cfg for polygen training
    """
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]

    ##### prepare data #####
    t = Compose(
        [
            QuantizeVertices8Bit(
                remove_duplicates=True, scale=cfg["data"]["quantize_scale"]
            ),
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
        cfg["data"]["data_path"],
        cfg["data"]["train_list"],
        multi_objs=True,
        classes_file=cfg["data"]["classes_file"],
        transform=t,
    )
    val_set = PLYDataset(
        cfg["data"]["data_path"],
        cfg["data"]["val_list"],
        multi_objs=True,
        classes_file=cfg["data"]["classes_file"],
        transform=t,
    )

    if subset:
        n = 128
        print(f"Running on {n} samples only!")
        train_set = Subset(train_set, range(n))
        val_set = Subset(val_set, range(n))

    print(f"Train set: {len(train_set)}")
    print(f"Val set: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["vtx"]["batch_size"],
        collate_fn=plydataset_collate_fn,
        num_workers=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["vtx"]["val_batch_size"],
        collate_fn=plydataset_collate_fn,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )
    ##### done preparing dataset ######
    num_classes = get_num_classes(cfg["data"]["classes_file"])
    print(f"Class conditional model with {num_classes} classes")

    ckpt = cfg["train"]["vtx"]["resume"]
    # if we have a checkpoint, load the model from that
    # the model params might have changed in the config
    if ckpt is not None:
        print(f"Resume from checkpoint: {ckpt}")
        model = VertexModel.load_from_checkpoint(ckpt)
    else:
        # read the model params from the config
        print("Train from scratch")
        model = VertexModel(vtx_cfg, cfg["train"], num_classes=num_classes)

    print(f"Model parameters: {count_parameters(model)}")

    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=5,)

    trainer = Trainer(
        gpus=1,
        gradient_clip_val=1.0,
        max_epochs=cfg["train"]["epochs"],
        checkpoint_callback=checkpoint_callback,
        fast_dev_run=dev,
        resume_from_checkpoint=ckpt,
        val_check_interval=cfg["train"]["vtx"]["val_intv"],
    )
    # allow logging dict
    # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/665
    monkeypatch_tensorboardlogger(trainer.logger)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    parser.add_argument(
        "--subset", action="store_true", help="Quick training on a subset of the data"
    )
    parser.add_argument("--dev", action="store_true", help="Train only 1 batch")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_training(cfg, args.subset, args.dev)
