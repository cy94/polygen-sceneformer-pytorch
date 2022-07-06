import argparse
from datasets.suncg_joint_dataset_deepsynth import SUNCG_Dataset
from torch.utils.data import DataLoader, Subset
from utils.config import read_config
from torchvision.transforms import Compose
from transforms.scene import SeqToTensor, Padding, Padding_joint  # , get_relation
from models.scene_joint import scene_transformer
from models.misc import count_parameters
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
import copy

seed_everything(1)


def log_metrics(self, metrics, step=None):
    for k, v in metrics.items():
        if isinstance(v, dict):
            self.experiment.add_scalars(k, v, step)
        else:
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.experiment.add_scalar(k, v, step)


def monkeypatch_tensorboardlogger(cfg, logger):
    import types

    logger.log_metrics = types.MethodType(log_metrics, logger)


def run_training(cfg, subset):

    t = Compose([Padding_joint(cfg), SeqToTensor()])

    trainval_set = SUNCG_Dataset(
        data_folder=cfg["data"]["data_path"],
        list_path=cfg["data"]["list_path"],
        transform=t,
    )
    total_len = cfg["train"]["total_len"]
    train_len = cfg["train"]["train_len"]

    # randomly split the data
    # select a fixed number of samples from the dataset
    train_set = Subset(trainval_set, range(1, train_len + 1))
    train_set.dataset = copy.deepcopy(trainval_set)
    train_set.dataset.augmentation = True
    val_set = Subset(trainval_set, range(train_len + 1, total_len + 1))

    if subset:
        n = 128
        print(f"Running on {n} samples only!")
        train_set = Subset(train_set, range(n))
        val_set = Subset(val_set, range(n))

    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=8
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["train"]["batch_size"], num_workers=8
    )

    model = scene_transformer(cfg)

    print(f"Model parameters: {count_parameters(model)}")

    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=5)

    trainer = Trainer(
        gpus=0,
        gradient_clip_val=1.0,
        max_epochs=cfg["train"]["epochs"],
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=cfg["train"]["resume"],
    )

    monkeypatch_tensorboardlogger(cfg, trainer.logger)

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    parser.add_argument(
        "--subset", action="store_true", help="Quick training on a subset of the data"
    )
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_training(cfg, args.subset)
