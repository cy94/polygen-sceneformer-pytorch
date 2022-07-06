from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.optim import Adam
import torch.nn.functional as F

from models.resnet import resnet34


class ResNetClassifier(LightningModule):
    def __init__(self, num_input_channels=37):
        super().__init__()
        self.model = resnet34(pretrained=False, num_input_channels=num_input_channels)

    def configure_optimizers(self):
        self.optim = Adam(self.parameters(), lr=3e-4, weight_decay=1e-1)

        return self.optim

    def validation_step(self, batch, batch_idx):
        loss, acc = self.eval_common(batch)
        x, y = batch
        # number of correct predictions
        correct = acc * len(y)
        samples = torch.Tensor([len(y)])

        return {"val_loss": loss, "correct": correct, "samples": samples}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        total_correct = torch.stack([x["correct"] for x in outputs]).sum()
        total_samples = torch.stack([x["samples"] for x in outputs]).sum()

        avg_acc = total_correct / total_samples

        log = {
            "loss": {"val": avg_loss},
            "acc": {"val": avg_acc},
        }
        return {"val_loss": avg_loss, "log": log}

    def eval_common(self, batch):
        x, y = batch
        preds = self(x)
        # CE loss even though its 2 classes, not BCE loss
        loss = F.cross_entropy(preds, y)
        acc = (preds.argmax(-1) == y).float().sum() / len(y)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.eval_common(batch)

        log = {"loss": {"train": loss}, "acc": {"train": acc}}

        return {"loss": loss, "log": log}

    def forward(self, x):
        return self.model(x)
