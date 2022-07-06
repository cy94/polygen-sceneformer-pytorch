import argparse

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything

from models.resnet_classifier import ResNetClassifier
from utils.config import read_config
from utils.log import monkeypatch_tensorboardlogger
from datasets.composite_dataset import CompositeDataset

seed_everything(1)


def train(cfg):
    train_set = CompositeDataset(
        cfg["data"]["train"]["gt"], cfg["data"]["train"]["preds"]
    )
    val_set = CompositeDataset(cfg["data"]["val"]["gt"], cfg["data"]["val"]["preds"])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=16)

    model = ResNetClassifier(num_input_channels=cfg["model"]["num_input_channels"])

    trainer = Trainer(
        gpus=1,
        max_epochs=100,
        checkpoint_callback=False,
        val_check_interval=10,
        log_save_interval=10,
        row_log_interval=10,
    )
    monkeypatch_tensorboardlogger(trainer.logger)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    train(cfg)
