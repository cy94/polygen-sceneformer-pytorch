import os.path as osp

import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.nn import PointConv, fps, global_max_pool, radius
from torch.utils.tensorboard import SummaryWriter


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
        ]
    )


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 10)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


def step(model, loaders, optimizer=None):
    losses = {}
    accs = {}

    for split in loaders:
        if split == "train":
            model.train()
        else:
            model.eval()

        loader = loaders[split]
        correct = 0
        full_loss = 0

        for batch, data in enumerate(tqdm(loader)):
            data = data.to(device)

            if split == "train":
                optimizer.zero_grad()

            # compute gradients only during train
            with torch.set_grad_enabled(split == "train"):
                # forward pass
                logits = model(data)

                # calculate loss and accuracy
                loss = F.nll_loss(logits, data.y)
                full_loss += loss.item()
                preds = logits.max(1)[1]
                correct += preds.eq(data.y).sum().item()

            if split == "train":
                loss.backward()
                optimizer.step()

        accs[split] = correct / len(loader.dataset)
        losses[split] = full_loss / len(loader.dataset)

    return losses, accs


if __name__ == "__main__":
    with open("pointnet2_classification.yaml") as f:
        cfg = yaml.safe_load(f)
    ##
    # load dataset
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    trainval_set = ModelNet(cfg["data_path"], "10", True, transform, pre_transform)
    test_set = ModelNet(cfg["data_path"], "10", False, transform, pre_transform)
    ##
    # split trainval into train and val
    train_len = int(0.8 * len(trainval_set))
    val_len = len(trainval_set) - train_len
    train_set, val_set = random_split(trainval_set, [train_len, val_len])

    # select only some of the train_data
    train_set = train_set[: cfg["train"]["select_data"]]
    val_set = val_set[: cfg["train"]["select_data"]]
    ### create data loaders
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    train_loaders = {"train": train_loader, "val": val_loader}
    test_loaders = {"test": test_loader}

    writer = SummaryWriter("PN2classification/v2")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        losses, accs = step(model, train_loaders, optimizer)

        tloss, vloss = losses["train"], losses["val"]
        tacc, vacc = accs["train"], accs["val"]

        writer.add_scalar("Loss/train", tloss, epoch)
        writer.add_scalar("Loss/val", vloss, epoch)
        writer.add_scalar("Accuracy/train", tacc, epoch)
        writer.add_scalar("Accuracy/val", vacc, epoch)

        print(
            f"Epoch: {epoch}\tTacc: {tacc}\tTloss: {tloss}\tVacc: {vacc}\tVloss: {vloss}"
        )

    losses, accs = step(model, test_loaders)
    test_acc = accs["test"]
    print(f"Test acc: {test_acc}")
