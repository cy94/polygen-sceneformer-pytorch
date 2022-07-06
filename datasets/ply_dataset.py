import os, os.path as osp
from pathlib import Path
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from plyfile import PlyData


def plydataset_collate_fn(batch):
    collated_batch = {
        # lists
        "vertices_raw": [mesh["vertices_raw"] for mesh in batch],
        "faces_raw": [mesh["faces_raw"] for mesh in batch],
        # tensors
        "vertices": torch.stack([mesh["vertices"] for mesh in batch], dim=0),
        "faces": torch.stack([mesh["faces"] for mesh in batch], dim=0),
        "face_pos": torch.stack([mesh["face_pos"] for mesh in batch], dim=0),
        "in_face_pos": torch.stack([mesh["in_face_pos"] for mesh in batch], dim=0),
    }
    # should we stack  the class indices?
    if "class_ndx" in batch[0]:
        collated_batch["class_ndx"] = torch.LongTensor(
            [mesh["class_ndx"] for mesh in batch]
        )

    return collated_batch


def gen_ply(ply_path, vertices, faces=None):
    """
    path: path to PLY file
    vertices: (n, 3) array
    """
    path = Path(ply_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        if vertices is not None:
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
        if faces is not None:
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        if vertices is not None:
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
        if faces is not None:
            for face in faces:
                f.write(f"{len(face)} ")
                for i in face:
                    f.write(f"{i} ")
                f.write("\n")


def get_num_classes(fpath):
    return len(read_file_lines(fpath))


def read_file_lines(fpath):
    with Path(fpath).open() as f:
        lines = f.readlines()
        # remove newline from each filename
        clean_lines = list(map(lambda s: s.rstrip(), lines))

    return clean_lines


class PLYDataset(Dataset):
    def __init__(
        self,
        data_folder,
        list_path=None,
        transform=None,
        multi_objs=False,
        classes_file=None,
    ):
        """
        data_folder: folder containing ply files
        list_path: (optional) path to list file containing filenames to read
        transform: (optional) transform to apply on the samples 
        multi_objs: the data_folder contains multiple object directories (each corresponding to a class ID)
        classes_file: list of ShapeNet class IDs
        """

        self.root = data_folder
        self.multi_objs = multi_objs
        self.classes = None

        if classes_file:
            self.classes = read_file_lines(classes_file)

        # paths relative to root are specified
        if list_path:
            self.files = read_file_lines(list_path)
        # look inside each object directory
        elif multi_objs:
            self.files = []
            # get the subfolders in the root folder
            subfolders = next(os.walk(self.root))[1]

            # in each subfolder, list all the meshes
            for object_dir in subfolders:
                # add all the PLY files in this subfolder to the list
                self.files += [
                    osp.join(object_dir, f)
                    for f in os.listdir(osp.join(data_folder, object_dir))
                    if f.endswith(".ply")
                ]
        # everything is in a single folder
        else:
            all_files = os.listdir(self.root)
            self.files = list(filter(lambda s: s.endswith(".ply"), all_files))

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        """
        Read the i-th PLY from file
        """
        path = osp.join(self.root, self.files[i])
        plydata = PlyData.read(path)

        vertices = np.array(plydata["vertex"].data.tolist())
        faces = np.array([f[0] for f in plydata["face"].data])

        sample = {"vertices": vertices, "faces": faces, "filename": self.files[i]}

        # add the class index in the multi class setting
        if self.classes:
            class_name = str(Path(sample["filename"]).parent)
            sample["class_ndx"] = self.classes.index(str(class_name))

        if self.transform:
            sample = self.transform(sample)

        return sample
