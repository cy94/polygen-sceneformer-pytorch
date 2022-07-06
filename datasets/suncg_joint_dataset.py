import os, os.path as osp
from torch.utils.data import Dataset
import pickle
import csv
import json
import numpy as np
import pandas as pd
import transforms3d
from datasets.filter import GlobalCategoryFilter
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import torch
from torchvision.transforms import Compose
from transforms.scene import SeqToTensor, Padding_joint
import yaml

path = "tests/configs/scene_joint_config.yaml"
with open(path) as f:
    cfg = yaml.safe_load(f)

t = Compose([Padding_joint(cfg), SeqToTensor()])


class SUNCG_Dataset(Dataset):
    def __init__(
        self, data_folder="datasets/suncg_bedroom/", list_path=None, transform=t
    ):
        self.root = data_folder
        if list_path:
            with Path(list_path).open() as f:
                file_list = f.readlines()
                self.files = list(map(lambda s: s.rstrip(), file_list))

        else:
            all_files = os.listdir(self.root)
            self.files = list(filter(lambda s: s.endswith(".pkl"), all_files))
        model_cat_file = "datasets/ModelCategoryMapping.csv"
        # df_Model = pd.read_csv(model_cat_file)
        # # get all model categories
        # self.all_cat = df_Model['fine_grained_class'].unique()[5:]
        # get model_to_categories, copy from fast_synth
        self.model_to_categories = {}
        with open(model_cat_file, "r") as f:
            categories = csv.reader(f)
            for l in categories:
                self.model_to_categories[l[1]] = [l[2], l[3], l[5]]
        # get cat req from file
        self.cat_freq = {}
        with open("datasets/room_model_fine_frequency", "r") as f:
            self.cat_freq = json.load(f)

        self.transform = transform
        self.bedroom_filter = GlobalCategoryFilter.get_bedroom_filter()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path = osp.join(self.root, self.files[item])
        with open(path, "rb") as f:
            room_dic = pickle.load(f)

        cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq = self.get_info(room_dic)

        sample = {
            "cat_seq": cat_seq,
            "x_loc_seq": x_loc_seq,
            "y_loc_seq": y_loc_seq,
            "z_loc_seq": z_loc_seq,
            "orient_seq": orient_seq,
        }
        # seq le , 长度咋弄？

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_info(self, dic):
        room = dic["room"]
        room_type = room["roomTypes"]
        objects = dic["objects"]
        objects_sorted = self.sort_filter_obj(objects, room_type)
        room_dim = np.array(room["bbox"]["max"]) - np.array(room["bbox"]["min"])
        room_size = np.sqrt(sum(room_dim ** 2))
        cat_seq = np.array([])
        x_loc_seq = np.array([])
        y_loc_seq = np.array([])
        z_loc_seq = np.array([])
        loc_seq = np.array([])
        orient_seq = np.array([])
        filtered, rejected, door_window = GlobalCategoryFilter.get_filter()

        for obj in objects_sorted:
            cat = self.model_to_categories[obj["modelId"]][0]
            if cat in self.bedroom_filter:
                cat_idx = np.where(np.array(self.bedroom_filter) == cat)[0]
                trans = np.array(obj["transform"]).reshape(4, 4).T
                # get x, y, z coordinates of objects
                loc = (
                    transforms3d.affines.decompose44(trans)[0]
                    - np.array(room["bbox"]["min"])
                ) / room_size
                # get rotation degree
                rot_matrix = transforms3d.affines.decompose44(trans)[1]
                r = R.from_matrix(rot_matrix)
                orient = r.as_euler("yzx", degrees=True)[0]
                # scale
                loc, orient = self.scale(loc=loc, orient=orient)
                x, y, z = loc[0], loc[1], loc[2]

                # orient = np.array([rot_matrix[0, 0], rot_matrix[0, 2]])
                # dim = (np.array(obj['bbox']['max']) - np.array(obj['bbox']['min'])) / room_size
                # scale up the loc, orient, dim
                # get object sequence
                # object_seq = np.concatenate((cat_idx.reshape(1, 1)))#, loc))#, orient, dim))
                # get scene sequence
                # if (object_seq >= 0).all() == True:
                # cat_idx_repeat = np.repeat(cat_idx, 3)

                cat_seq = np.hstack((cat_seq, cat_idx))
                # loc_seq = np.hstack((loc_seq, loc))
                orient_seq = np.hstack((orient_seq, orient))
                x_loc_seq = np.hstack((x_loc_seq, x))
                y_loc_seq = np.hstack((y_loc_seq, y))
                z_loc_seq = np.hstack((z_loc_seq, z))

        return cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq

    def sort_filter_obj(self, objects, room_type):
        index_freq_pairs = []
        for index in range(len(objects)):
            obj = objects[index]
            if "modelId" in obj.keys() and obj["transform"][0] != None:
                # using the fine cat, can be changed to coarse
                cat = self.model_to_categories[obj["modelId"]][0]
                # get model freq based on room_type and model cat
                freq = self.cat_freq[repr(room_type)][cat]
                # set the index to frequency pairs
                index_freq_pairs.append((index, freq))
                index_freq_pairs.sort(key=lambda tup: tup[1], reverse=True)
        # sort objects based on freq
        sorted_objects = [objects[tup[0]] for tup in index_freq_pairs]
        return sorted_objects

    def scale(self, loc, orient=None, dim=None):
        loc *= 200
        if orient is not None:
            orient += 180
            # orient *= 100
        if dim is not None:
            dim *= 200
        return loc, orient  # , dim


# dataset = SUNCG_Dataset()
# dataset[3]
# samples_seq = []
# i = 0
