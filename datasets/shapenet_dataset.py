import numpy as np

from torch.utils.data import Dataset
from kaolin.datasets import shapenet


class ShapeNetDataset(Dataset):
    def __init__(self, root, categories=None, transform=None):
        self.transform = transform
        self._dataset = shapenet.ShapeNet_Meshes(root=root, categories=categories)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        mesh = self._dataset[i]
        vertices, faces = mesh["data"]["vertices"], mesh["data"]["faces"]
        sample = {"vertices": np.array(vertices), "faces": np.array(faces).tolist()}

        if self.transform:
            sample = self.transform(sample)

        return sample
