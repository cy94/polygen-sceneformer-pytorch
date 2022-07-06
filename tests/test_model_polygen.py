from pathlib import Path
import numpy as np
from datasets.ply_dataset import PLYDataset, plydataset_collate_fn, gen_ply
from transforms.polygen import (
    SortMesh,
    MeshToSeq,
    QuantizeVertices8Bit,
    GetFacePositions,
    MeshToTensor,
)
from utils.config import read_config
from models.polygen import (
    PolyGen,
    generate_square_subsequent_mask,
    VertexModel,
    FaceModel,
)
from torch.utils.data import random_split
import torch
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pytest
from tqdm import tqdm

DATA_PATHS = ["tests/data/table"]

DATA_LISTS = [
    "tests/data/lists/table.txt",
]

CONFIGS = ["tests/configs/polygen.yaml"]


def test_square_mask():
    mask = generate_square_subsequent_mask(10)
    assert mask.shape[0] == mask.shape[1] == 10


def split_list(input_list, seperator):
    outer = []
    inner = []
    for elem in input_list:
        if elem == seperator:
            if inner:
                outer.append(inner)
            inner = []
        else:
            inner.append(elem)
    if inner:
        outer.append(inner)
    return outer


@pytest.mark.parametrize(
    "data_path, list_path, cfg_path", zip(DATA_PATHS, DATA_LISTS, CONFIGS)
)
def test_vertex_model(data_path, list_path, cfg_path):
    cfg = read_config(cfg_path)
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]
    t = Compose(
        [
            QuantizeVertices8Bit(remove_duplicates=True),
            SortMesh(),
            GetFacePositions(new_face_token=face_cfg["new_face_token"]),
            MeshToSeq(
                vtx_start_token=vtx_cfg["vertex_start_token"],
                vtx_stop_token=vtx_cfg["vertex_stop_token"],
                vtx_pad_token=vtx_cfg["vertex_pad_token"],
                vtx_seq_len=vtx_cfg["max_seq_len"],
                new_face_token=-1,
            ),
            MeshToTensor(),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vtx_stop_token = vtx_cfg["vertex_stop_token"]

    dataset = PLYDataset(data_path, list_path=list_path, transform=t)
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], collate_fn=plydataset_collate_fn
    )
    good_outputs = []
    # run 5 training experiments
    for i in range(1):
        losses = []
        model = VertexModel(cfg=vtx_cfg).to(device)
        optim = Adam(model.parameters())
        model.train()
        writer = SummaryWriter("tensorboard")
        for epoch in range(500):
            epoch_loss = 0

            for batch in loader:
                optim.zero_grad()
                vertices, faces = batch["vertices"], batch["faces"]
                vertices = torch.LongTensor(vertices).to(device)
                out = model(vertices)
                batch_loss = 0

                for ndx in range(out.shape[0]):
                    preds = out[ndx]
                    gt = vertices[ndx].clone()
                    # change the stop token into its class index (subtract 1)
                    gt[gt == vtx_stop_token] -= 1
                    # remove start tokens from gt - we dont want to predict these finally
                    gt = gt[1:]
                    # TODO: ignore padding
                    loss = F.nll_loss(preds, gt)
                    batch_loss += loss
                writer.add_scalar("Loss/train", batch_loss, epoch)
                batch_loss /= len(batch)
                batch_loss.backward()
                optim.step()

                epoch_loss += batch_loss.detach().cpu().numpy()
            losses.append(epoch_loss)
            print(f"Epoch: {epoch}, Loss: {epoch_loss:.4f}")
        # final loss should be lesser than the initial loss
        assert losses[-1] < losses[0]
        # generate a single new sequence (only one possible with
        # an unconditional model)
        model.eval()
        with torch.no_grad():
            vertices = model.greedy_decode().numpy()
            # start token is always in the output
            # check for stop token and vertices occuring in zyx triplets
            if vtx_stop_token in vertices and (len(vertices) - 2) % 3 == 0:
                good_outputs.append(vertices[1:-1])

    assert len(good_outputs) > 0
    for (ndx, vertices) in enumerate(good_outputs):
        ply_path = Path(cfg["data"]["out_path"]) / f"out_{ndx}.ply"
        vertices = vertices.reshape((-1, 3))[:, ::-1]
        gen_ply(ply_path, vertices, faces=None)


@pytest.mark.parametrize(
    "data_path, list_path, cfg_path", zip(DATA_PATHS, DATA_LISTS, CONFIGS)
)
def test_face_model(data_path, list_path, cfg_path):
    cfg = read_config(cfg_path)
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]

    t = Compose(
        [
            QuantizeVertices8Bit(remove_duplicates=True),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PLYDataset(data_path, list_path=list_path, transform=t)
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], collate_fn=plydataset_collate_fn
    )
    losses = []

    model = FaceModel(face_cfg, vtx_cfg).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    model.train()
    writer = SummaryWriter("tensorboard")
    for epoch in range(800):
        epoch_loss = 0

        for batch_ndx, batch in enumerate(loader):
            optim.zero_grad()
            vertices_raw, faces_raw = batch["vertices_raw"], batch["faces_raw"]
            vertices, faces = batch["vertices"], batch["faces"]
            face_pos, in_face_pos = batch["face_pos"], batch["in_face_pos"]

            vertices_raw = torch.LongTensor(vertices_raw).to(device)
            vertices = torch.LongTensor(vertices).to(device)
            faces = torch.LongTensor(faces).to(device)

            face_pos = torch.LongTensor(face_pos).to(device)
            in_face_pos = torch.LongTensor(in_face_pos).to(device)

            if epoch == 0 and batch_ndx == 0:
                print("raw vertices", vertices_raw)
                print("raw faces", faces_raw)
                print("Input vertices", vertices)
                print("Input faces", faces)
                print("facepos", face_pos)
                print("in_face_pos", in_face_pos)

            out = model(face_pos, in_face_pos, faces, vertices, vertices_raw)
            batch_loss = 0

            for ndx, _ in enumerate(out):
                preds = out[ndx]

                gt = faces[ndx].clone()
                # remove start tokens from gt - we dont want to predict these finally
                gt = gt[1:]
                # -1 score is the stop face token
                gt[gt == face_cfg["face_stop_token"]] = preds.shape[1] - 1
                # -2 score is the new face token
                gt[gt == face_cfg["new_face_token"]] = preds.shape[1] - 2

                loss = F.nll_loss(preds, gt, ignore_index=face_cfg["face_pad_token"])
                batch_loss += loss

            batch_loss /= len(batch)
            writer.add_scalar("Loss/train", batch_loss, epoch)
            batch_loss.backward()
            optim.step()

            epoch_loss += batch_loss.detach().cpu().numpy()
        losses.append(epoch_loss)
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss:.8f}")

    assert losses[-1] < losses[0]

    # generate a single output mesh
    model.eval()
    with torch.no_grad():
        for batch_ndx, batch in enumerate(loader):
            # get GT vertices of the first mesh in the batch
            vertices_raw = torch.LongTensor(batch["vertices_raw"]).to(device)[0]
            out_faces = model.greedy_decode(vertices_raw)
            print(out_faces)
            # write into PLY file
            ply_path = Path(cfg["data"]["out_path"]) / f"out_{0}.ply"
            split_output = split_list(out_faces[:-1], 256)
            gen_ply(ply_path, batch["vertices_raw"][0], faces=split_output)
            break


@pytest.mark.parametrize(
    "data_path, list_path, cfg_path", zip(DATA_PATHS, DATA_LISTS, CONFIGS)
)
def test_eval_vertex_model(data_path, list_path, cfg_path):
    cfg = read_config(cfg_path)
    vtx_cfg = cfg["model"]["vertex_model"]
    face_cfg = cfg["model"]["face_model"]
    t = Compose(
        [
            QuantizeVertices8Bit(remove_duplicates=True),
            SortMesh(),
            GetFacePositions(new_face_token=face_cfg["new_face_token"]),
            MeshToSeq(
                vtx_start_token=vtx_cfg["vertex_start_token"],
                vtx_stop_token=vtx_cfg["vertex_stop_token"],
                vtx_pad_token=vtx_cfg["vertex_pad_token"],
                vtx_seq_len=vtx_cfg["max_seq_len"],
                new_face_token=-1,
            ),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vtx_stop_token = vtx_cfg["vertex_stop_token"]

    trainval_set = PLYDataset(data_path, list_path=list_path, transform=t)
    train_len = int(0.8 * len(trainval_set))
    val_len = len(trainval_set) - train_len
    train_set, val_set = random_split(trainval_set, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        collate_fn=plydataset_collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["train"]["batch_size"], collate_fn=plydataset_collate_fn
    )
    loaders = {"train": train_loader, "val": val_loader}
    good_outputs = []
    # run 5 training experiments
    # losses = []
    model = VertexModel(cfg=vtx_cfg).to(device)
    optim = Adam(model.parameters())
    model.train()
    writer_train = SummaryWriter("tensorboard/run2/train")
    writer_val = SummaryWriter("tensorboard/run2/val")
    for epoch in range(80):
        epoch_loss = 0
        epoch_eval = 0

        for split in loaders:
            if split == "train":
                model.train()
            else:
                model.eval()
            loader = loaders[split]

            for batch in loader:
                if split == "train":
                    optim.zero_grad()
                vertices, faces = batch["vertices"], batch["faces"]
                vertices = torch.LongTensor(vertices).to(device)

                with torch.set_grad_enabled(split == "train"):
                    out = model(vertices)
                    batch_loss = 0

                    for ndx in range(out.shape[0]):
                        preds = out[ndx]
                        gt = vertices[ndx].clone()
                        # change the stop token into its class index (subtract 1)
                        gt[gt == vtx_stop_token] -= 1
                        # remove start tokens from gt - we dont want to predict these finally
                        gt = gt[1:]
                        # TODO: ignore padding
                        loss = F.nll_loss(preds, gt)
                        batch_loss += loss
                # batch_loss /= len(batch)
                if split == "train":
                    batch_loss.backward()
                    optim.step()
                    epoch_loss += batch_loss.detach().cpu().numpy()
                else:
                    epoch_eval += batch_loss.detach().cpu().numpy()
            # losses.append(epoch_loss)

        epoch_loss = epoch_loss / len(loaders["train"].dataset)
        epoch_eval = epoch_eval / len(loaders["val"].dataset)
        print(f"Epoch: {epoch}, Train: {epoch_loss:.4f}, Val: {epoch_eval:.4f}")
        writer_train.add_scalar("Loss", epoch_loss, epoch)
        writer_val.add_scalar("Loss", epoch_eval, epoch)
    # final loss should be lesser than the initial loss
    # assert losses[-1] < losses[0]
    # generate a single new sequence (only one possible with
    # an unconditional model)
    model.eval()
    with torch.no_grad():
        vertices = model.greedy_decode().numpy()
        # start token is always in the output
        # check for stop token and vertices occuring in zyx triplets
        if vtx_stop_token in vertices and (len(vertices) - 2) % 3 == 0:
            good_outputs.append(vertices[1:-1])

    # assert len(good_outputs) > 0
    for (ndx, vertices) in enumerate(good_outputs):
        ply_path = Path(cfg["data"]["out_path"]) / f"out_{ndx}.ply"
        vertices = vertices.reshape((-1, 3))[:, ::-1]
        gen_ply(ply_path, vertices, faces=None)
