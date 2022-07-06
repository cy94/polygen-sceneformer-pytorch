import json
from copy import deepcopy
import os
import os.path as osp
import yaml
import pickle
from pathlib import Path

from separate_models.scene_shift_cat import scene_transformer as cat_model_pre
from separate_models.scene_shift_ori_col import scene_transformer as ori_model_pre
from separate_models.scene_shift_loc_col import scene_transformer as loc_model_pre

import torch
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d
import trimesh

from tqdm import tqdm
from data.create_images import create_floor, create_wall
from data.top_down import TopDownView
from data.house import Room, Node
from data.rendered import RenderedScene
from data.dataset import toimage

####### Globals #########
path_cat = "lightning_logs/version_459/hparams.yaml"
path_loc = "/home/chandan/xplogs/version_307/hparams.yaml"
path_ori = "/home/chandan/xplogs/version_311/hparams.yaml"

with open(path_loc) as f:
    cfg = yaml.safe_load(f)

MAX_ROOM_SIZE = [6.05, 4.05, 6.05]

IMG_OUT_DIR = "/home/chandan/suncg/topdown/"
os.makedirs(osp.join(IMG_OUT_DIR, "gt"), exist_ok=True)
os.makedirs(osp.join(IMG_OUT_DIR, "preds"), exist_ok=True)

RENDER_OUT_DIR = "/home/chandan/suncg/composite/"
os.makedirs(osp.join(RENDER_OUT_DIR, "gt"), exist_ok=True)
os.makedirs(osp.join(RENDER_OUT_DIR, "preds"), exist_ok=True)

SEQ_OUT_DIR = "/home/chandan/suncg/seq"
os.makedirs(osp.join(SEQ_OUT_DIR, "gt"), exist_ok=True)
os.makedirs(osp.join(SEQ_OUT_DIR, "preds"), exist_ok=True)

SCENE_OUT_DIR = cfg["test"]["log_dir"]
os.makedirs(osp.join(SCENE_OUT_DIR, "gt"), exist_ok=True)
os.makedirs(osp.join(SCENE_OUT_DIR, "preds"), exist_ok=True)
print("Output dir", SCENE_OUT_DIR)

N_PREDS = 1200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_prepare(model, path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    model = model(cfg)
    model_path = osp.join(cfg["test"]["model_file"])
    state_dict = torch.load(model_path)["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    cat_model = model_prepare(cat_model_pre, path_cat)
    loc_model = model_prepare(loc_model_pre, path_loc)
    ori_model = model_prepare(ori_model_pre, path_ori)

with open("datasets/final_categories_frequency", "r") as f:
    cat_freq = json.load(f)
cat_list = list(cat_freq.keys())
model_cat_file = "datasets/ModelCategoryMapping.csv"
df = pd.read_csv(model_cat_file)
obj_path_root = "/shared/data/suncg/object"
####### End Globals #########


def update(gen_seq, next_token):
    """
    Add a new token to an existing sequence
    """
    gen_seq.append(next_token.item())
    return gen_seq


def update_repeat(repeat_gen_seq, next_token):
    """
    Add a new token 3 times to an existing sequence
    """
    for i in range(3):
        repeat_gen_seq.append(next_token.item())
    return repeat_gen_seq


def vals_to_node(model_id, transform, mesh):
    """
    Create a Node object from the given values 
    """
    # get the bbox of this object
    bbox = {}
    bbox["min"], bbox["max"] = mesh.bounds

    transform_flat = transform.T.reshape((-1)).tolist()

    # all the values required in the Node
    node_dict = {
        "parent": None,
        "child": [],
        "bbox": bbox,
        # column major format
        "transform": transform_flat,
        "modelId": model_id,
        "type": "Object",
        "valid": 1,
    }
    # level is not used
    node = Node(node_dict, level=None)
    return node


class MyObject(object):
    pass


def create_room(nodes):
    """
    nodes: list of Node objects contained in this room

    return: Room object
    """
    # create a fake level object
    level = MyObject()
    level.house = MyObject()
    level.house.id = "0004d52d1aeeb8ae6de39d6bd993e992"

    # create a Node to represent the room
    room_dict = {
        "bbox": {"min": [0, 0, 0], "max": MAX_ROOM_SIZE},
        "child": [],
        "house_id": "empty",
        "id": "1",
        "modelId": "fr_0rm_2",
        "nodeIndices": list(range(len(nodes))),
        "nodes": [],
        "roomTypes": ["Bedroom"],
        "type": "Room",
    }

    room_node = Node(room_dict, level=None)
    # now create a Room object with the same info
    r = Room(room=room_node, nodes=nodes, level=level)
    return r


def scene_gen(
    cat_gen_seq,
    x_gen_seq,
    y_gen_seq,
    z_gen_seq,
    orient_gen_seq,
    ndx,
    out_dir="out",
    modelids=None,
):
    """
    Generate GLB mesh files for each scene
    Generate a deep-synth pickled "Room" object for each scene
    This can later be used to render the TopDownView of the generated scene

    out_dir: 'gt', 'preds' - the suffix added to the base dir
    """
    # remove the start tokens, un-normalize values
    cat = np.array(cat_gen_seq[1:])
    x = np.array(x_gen_seq[1:]) * 6.05 * 1.5 / 200 - 6.05 / 4
    y = np.array(y_gen_seq[1:]) * 4.05 * 1.5 / 200 - 4.05 / 4
    z = np.array(z_gen_seq[1:]) * 6.05 * 1.5 / 200 - 6.05 / 4
    orient = np.array(orient_gen_seq[1:]) - 180

    scene = trimesh.scene.scene.Scene()

    obj_nodes = []

    for obj_ndx, _ in enumerate(cat):
        if cat[obj_ndx] < len(cat_freq):
            # translation
            T = np.array([x[obj_ndx], y[obj_ndx], z[obj_ndx]])
            # rotation angle
            r = R.from_euler("y", orient[obj_ndx], degrees=True)
            # rotation matrix
            Rotation = r.as_matrix()
            Z = [1, 1, 1]
            # full transformation matrix
            transform = transforms3d.affines.compose(T, Rotation, Z, S=None)

            # modelids are given
            if modelids is not None:
                model_id = modelids[obj_ndx]
            # modelids not given, pick the first object in this category
            else:
                # the category name
                model_cat = cat_list[cat[obj_ndx]]
                # first model in this category
                model_id = df[df["fine_grained_class"] == model_cat][:1][
                    "model_id"
                ].item()

            # remove the "mirror" part of model id
            if "mirror" in model_id:
                # model_id = model_id.split('_')[0]
                model_id = model_id[:-7]
            # path to model
            obj_path = osp.join(obj_path_root, f"{model_id}/{model_id}.obj")
            # load the mesh for this object
            try:
                mesh = trimesh.load_mesh(obj_path)
            except ValueError:
                print("looking for model", modelids[obj_ndx])
            # transform it to the required location
            mesh.apply_transform(transform)

            scene.add_geometry(mesh)

            # create a "Node" object
            node = vals_to_node(model_id, transform, mesh)
            obj_nodes.append(node)

    scene_file = osp.join(SCENE_OUT_DIR, out_dir, f"{ndx}.glb")
    scene.export(scene_file)
    # now create a Room object with all the nodes
    room = create_room(obj_nodes)

    renderer = TopDownView(size=512)
    img, data = renderer.render(room)

    # replace the floor and wall with our "fake" floor and wall
    _, _, nodes = data
    fake_floor, fake_wall = create_floor(), create_wall()
    data = (create_floor(), create_wall(), nodes)

    # write the topdown 2d image to file
    img += fake_floor.numpy() + fake_wall.numpy()
    img_rgb = toimage(img, cmin=0, cmax=1)
    img_out_path = osp.join(IMG_OUT_DIR, out_dir, f"{ndx}.jpg")
    img_rgb.save(img_out_path)

    with open("datasets/0.pkl", "wb") as f:
        pickle.dump((data, room), f, pickle.HIGHEST_PROTOCOL)
    scene = RenderedScene(0, "datasets", ".")
    composite = scene.create_composite()
    try:
        composite.add_nodes(scene.object_nodes)
        composite_img = composite.get_composite(num_extra_channels=0, ablation="basic")
        out_file = f"{ndx}.pth"
        render_out_path = osp.join(RENDER_OUT_DIR, out_dir, out_file)
        torch.save(composite_img, render_out_path)
    except RuntimeError:
        print("Could not create composite")

    # save the raw sequences to file
    # useful for analysis like KL divergence of categories
    out_file = f"{ndx}.pth"
    seq_out_path = osp.join(SEQ_OUT_DIR, out_dir, out_file)
    torch.save(
        {
            "cat": np.array(cat),
            "x": np.array(x),
            "y": np.array(y),
            "z": np.array(z),
            "orient": np.array(orient),
        },
        seq_out_path,
    )


if __name__ == "__main__":
    print("Generating predictions and rendering")
    # Generate N scenes
    for out_scene_ndx in tqdm(range(N_PREDS)):
        cat_gen_seq = [cfg["model"]["cat"]["start_token"]]
        x_gen_seq = [cfg["model"]["coor"]["start_token"]]
        y_gen_seq = [cfg["model"]["coor"]["start_token"]]
        z_gen_seq = [cfg["model"]["coor"]["start_token"]]
        ori_gen_seq = [cfg["model"]["orient"]["start_token"]]

        # seq for loc model
        loc_gen_seq = [cfg["model"]["coor"]["start_token"]]
        cat_repeat_gen_seq = []
        ori_repeat_gen_seq = []

        decoder_seq_len = cfg["model"]["max_seq_len"]
        generate_seq_len = decoder_seq_len - 1

        for out_ndx in range(generate_seq_len):
            cat_next_token = cat_model.decode_multi_model(
                out_ndx, cat_gen_seq, x_gen_seq, y_gen_seq, z_gen_seq, ori_gen_seq
            )
            if cat_next_token == 999:
                break
            cat_gen_seq = update(cat_gen_seq, cat_next_token)
            cat_repeat_gen_seq = update_repeat(cat_repeat_gen_seq, cat_next_token)

            #
            ori = ori_model.decode_multi_model(
                out_ndx, cat_gen_seq[1:], x_gen_seq, y_gen_seq, z_gen_seq, ori_gen_seq
            )
            if ori == 999:
                break
            ori_gen_seq = update(ori_gen_seq, ori)
            ori_repeat_gen_seq = update_repeat(ori_repeat_gen_seq, ori)

            x = loc_model.decode_multi_model(
                3 * out_ndx, cat_repeat_gen_seq, loc_gen_seq, ori_repeat_gen_seq
            )
            if x == 999:
                break
            loc_gen_seq = update(loc_gen_seq, x)
            x_gen_seq = update(x_gen_seq, x)

            y = loc_model.decode_multi_model(
                3 * out_ndx + 1, cat_repeat_gen_seq, loc_gen_seq, ori_repeat_gen_seq
            )
            if y == 999:
                break
            loc_gen_seq = update(loc_gen_seq, y)
            y_gen_seq = update(y_gen_seq, y)

            z = loc_model.decode_multi_model(
                3 * out_ndx + 2, cat_repeat_gen_seq, loc_gen_seq, ori_repeat_gen_seq
            )
            if z == 999:
                break
            loc_gen_seq = update(loc_gen_seq, z)
            z_gen_seq = update(z_gen_seq, z)

        scene_gen(
            cat_gen_seq,
            x_gen_seq,
            y_gen_seq,
            z_gen_seq,
            ori_gen_seq,
            out_scene_ndx,
            out_dir="preds",
        )
