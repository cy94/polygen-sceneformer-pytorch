import numpy as np
import torch
import os
import os.path as osp
from models.scene_joint import scene_transformer
from tqdm import tqdm
import argparse
from utils.config import read_config
from datasets.suncg_joint_dataset_deepsynth import SUNCG_Dataset
import pandas as pd
from datasets.filter import GlobalCategoryFilter
from scipy.spatial.transform import Rotation as R
import transforms3d
import trimesh
from torchvision.transforms import Compose
from transforms.scene import SeqToTensor, Padding_joint
from torch.utils.data import DataLoader, Subset
import json


def scene_gen(cat_gen_seq, x_gen_seq, y_gen_seq, z_gen_seq, orient_gen_seq, ndx):
    # frequencies
    with open("datasets/final_categories_frequency", "r") as f:
        cat_freq = json.load(f)
    # list of object classes
    cat_list = list(cat_freq.keys())
    # mapping from category to model ID
    model_cat_file = "datasets/ModelCategoryMapping.csv"
    df = pd.read_csv(model_cat_file)

    obj_path_root = "/shared/data/suncg/object"

    # remove the start and stop tokens
    # and reverse the preprocessing to get the raw values
    cat = np.array(cat_gen_seq[1:-1])
    x = np.array(x_gen_seq[1:-1]) * 6.05 * 1.5 / 200
    y = np.array(y_gen_seq[1:-1]) * 4.05 * 1.5 / 200
    z = np.array(z_gen_seq[1:-1]) * 6.05 * 1.5 / 200
    orient = np.array(orient_gen_seq[1:-1]) - 180

    count = 0

    for i in range(len(cat)):
        # if its a valid category
        if cat[i] < len(cat_freq):
            # translation of the object
            T = np.array([x[i], y[i], z[i]])
            # orientation of the object as a rotation matrix
            r = R.from_euler(
                "y", orient[i].cpu(), degrees=True
            )  # .cpu(), degrees=True)
            Rotation = r.as_matrix()
            # no scaling
            Z = [1, 1, 1]
            # get the full transformation matrix
            transform = transforms3d.affines.compose(T, Rotation, Z, S=None)

            # the category id of this object
            model_cat = cat_list[cat[i]]
            # get the ID for this model
            model_id = df[df["fine_grained_class"] == model_cat][:1]["model_id"].item()
            # path to OBJ file for this model
            obj_path = f"{model_id}/{model_id}.obj"
            obj_path = osp.join(obj_path_root, obj_path)
            mesh = trimesh.load_mesh(obj_path)
            # apply the transformation
            mesh.apply_transform(transform)

            # create a new scene if we dont have one
            if count == 0:
                scene = trimesh.scene.scene.Scene(geometry=mesh)
            # otherwise add the mesh
            scene.add_geometry(mesh)
            count += 1

    out_path = cfg["test"]["log_dir"]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    scene_file = osp.join(out_path, f"{ndx}.glb")
    scene.export(scene_file)


def run_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = Compose([Padding_joint(cfg), SeqToTensor()])
    trainval_set = SUNCG_Dataset(
        data_folder=cfg["data"]["data_path"],
        list_path=cfg["data"]["list_path"],
        transform=t,
    )

    # load model
    model_path = osp.join(cfg["test"]["model_file"])
    state_dict = torch.load(model_path)["state_dict"]
    model = scene_transformer(cfg)
    model.load_state_dict(state_dict)
    # move to GPU and switch to eval mode
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for out_ndx in tqdm(range(cfg["test"]["num_samples"])):
            # generate the sequence of objects
            (
                cat_gen_seq,
                x_gen_seq,
                y_gen_seq,
                z_gen_seq,
                orient_gen_seq,
            ) = model.greedy_decode(
                probabilistic=cfg["test"]["probabilistic"],
                nucleus=cfg["test"]["nucleus"],
            )
            # generate a complete scene mesh from the sequence
            scene_gen(
                cat_gen_seq, x_gen_seq, y_gen_seq, z_gen_seq, orient_gen_seq, out_ndx
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_inference(cfg)
