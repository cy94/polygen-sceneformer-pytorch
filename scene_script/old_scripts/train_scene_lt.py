import argparse
from datasets.suncg_dataset import SUNCG_Dataset
from torch.utils.data import DataLoader, Subset
from utils.config import read_config
from torchvision.transforms import Compose
from transforms.scene import SeqToTensor, Padding
from models.scene_transformer import scene_transformer
from models.misc import count_parameters
from pytorch_lightning import Trainer, seed_everything
import torch
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


def run_training(cfg):

    t = Compose(
        [
            Padding(
                start_token=cfg["model"]["start_token"],
                stop_token=cfg["model"]["stop_token"],
                pad_token=cfg["model"]["pad_token"],
                max_seq_len=cfg["model"]["max_seq_len"],
            ),
            SeqToTensor(),
        ]
    )

    trainval_set = SUNCG_Dataset(
        data_folder=cfg["data"]["data_path"],
        list_path=cfg["data"]["list_path"],
        transform=t,
    )
    total_len = 2200
    train_len = 2000

    # randomly split the data
    # train_set, val_set = random_split(trainval_set,
    # [train_len, len(trainval_set) - train_len])
    # select a fixed number of samples from the dataset
    train_set = Subset(trainval_set, range(train_len))
    val_set = Subset(trainval_set, range(train_len, total_len))

    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"])

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

    monkeypatch_tensorboardlogger(trainer.logger)
    # trainer.logger.log_hyperparams()
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_training(cfg)
