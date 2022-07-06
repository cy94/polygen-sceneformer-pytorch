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
import copy
import random


class SUNCG_Dataset(Dataset):
    def __init__(
        self, data_folder="datasets/suncg_bedroom/", list_path=None, transform=None
    ):
        self.root = data_folder

        all_files = os.listdir(self.root)
        self.files = list(filter(lambda s: s.endswith(".pkl"), all_files))
        model_cat_file = "datasets/ModelCategoryMapping.csv"

        self.model_to_categories = {}
        with open(model_cat_file, "r") as f:
            categories = csv.reader(f)
            for l in categories:
                self.model_to_categories[l[1]] = [l[2], l[3], l[5]]
        # get cat req from file
        # self.cat_freq = {}
        # with open('datasets/room_model_fine_frequency', 'r') as f:
        #     self.cat_freq = json.load(f)

        self.transform = transform
        self.room_size_cap = [6.05, 4.05, 6.05]
        with open("datasets/final_categories_frequency", "r") as f:
            self.cat_freq = json.load(f)

        self.augmentation = False

        # self.bedroom_filter = GlobalCategoryFilter.get_bedroom_filter()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path = osp.join(self.root, f"{item}.pkl")
        with open(path, "rb") as f:
            room_dic = pickle.load(f)

        room = copy.deepcopy(room_dic[1])

        if self.augmentation == True:
            # degree_list = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
            degree_list = [0, 90, 180, 270]
            degree = random.choice(degree_list)
            # degree = random.randint(0, 359)
            room = self.augment_rot(room, degree)
            room = self.augment_jittering(room)

        cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq = self.get_info(room)
        #
        sample = {
            "cat_seq": cat_seq,
            "x_loc_seq": x_loc_seq,
            "y_loc_seq": y_loc_seq,
            "z_loc_seq": z_loc_seq,
            "orient_seq": orient_seq,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_info(self, room):
        room_type = room.roomTypes
        objects = room.nodes
        objects_sorted = self.sort_filter_obj(objects, room_type)
        # room_dim = np.array(room['bbox']['max']) - np.array(room['bbox']['min'])
        # room_size = np.sqrt(sum(room_dim ** 2))
        cat_seq = np.array([])
        x_loc_seq = np.array([])
        y_loc_seq = np.array([])
        z_loc_seq = np.array([])
        loc_seq = np.array([])
        orient_seq = np.array([])
        # filtered, rejected, door_window  = GlobalCategoryFilter.get_filter()

        for obj in objects_sorted:
            cat = self.get_final_category(obj.modelId)
            # if cat in self.bedroom_filter:

            cat_idx = np.where(np.array(list(self.cat_freq.keys())) == cat)[0]
            trans = np.array(obj.transform).reshape(4, 4).T
            # shift
            shift = -(
                np.array(room.bbox["min"]) * 0.5
                + np.array(room.bbox["max"]) * 0.5
                - np.array(self.room_size_cap) * 1.5 * 0.5
            )
            # get x, y, z coordinates of objects
            loc = transforms3d.affines.decompose44(trans)[0] + shift
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
            # if 'modelId' in obj.keys():
            # using the fine cat, can be changed to coarse
            cat = self.get_final_category(obj.modelId)
            # get model freq based on room_type and model cat
            freq = self.cat_freq[cat]
            # set the index to frequency pairs
            index_freq_pairs.append((index, freq))
            index_freq_pairs.sort(key=lambda tup: tup[1], reverse=True)
        # sort objects based on freq
        sorted_objects = [objects[tup[0]] for tup in index_freq_pairs]
        return sorted_objects

    def get_final_category(self, model_id):
        """
        Final categories used in the generated dataset
        Minor tweaks from fine categories
        """
        model_id = model_id.replace("_mirror", "")
        category = self.model_to_categories[model_id][0]
        if model_id == "199":
            category = "dressing_table_with_stool"
        if category == "nightstand":
            category = "stand"
        if category == "bookshelf":
            category = "shelving"
        return category

    def scale(self, loc, orient=None, dim=None):
        loc *= 200 / (np.array(self.room_size_cap) * 1.5)
        if orient is not None:
            orient += 180
            # orient *= 100
        if dim is not None:
            dim *= 200
        return loc, orient  # , dim

    def augment_rot(self, room, degree):
        # get transformation matrix
        r = R.from_euler("y", degree, degrees=True)
        ro = r.as_matrix()
        T = np.array([0, 0, 0])
        Z = [1, 1, 1]
        t_rot = transforms3d.affines.compose(T, ro, Z, S=None)

        def update_bbox(node):

            (xmin, zmin, ymin) = list(
                rotate(np.asarray([node.xmin, node.zmin, node.ymin, 1]))[0:3]
            )
            (xmax, zmax, ymax) = list(
                rotate(np.asarray([node.xmax, node.zmax, node.ymax, 1]))[0:3]
            )

            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            node.bbox["min"] = (xmin, zmin, ymin)
            node.bbox["max"] = (xmax, zmax, ymax)
            (node.xmin, node.zmin, node.ymin) = node.bbox["min"]
            (node.xmax, node.zmax, node.ymax) = node.bbox["max"]

        def rotate(t):
            t = np.dot(t, t_rot)  # np.linalg.matrix_power(t_rot,(i+1)))
            return t

        room.transform = t_rot
        for node in room.nodes:
            node.transform = list(
                rotate(np.asarray(node.transform).reshape(4, 4)).flatten()
            )
            update_bbox(node)
        update_bbox(room)

        return room

    def augment_jittering(self, room):
        jitter = np.random.uniform(low=0.0, high=0.5, size=3)
        for node in room.nodes:
            node.transform[-4:-1] += jitter

        return room
