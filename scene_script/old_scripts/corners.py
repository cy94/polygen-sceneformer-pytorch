import argparse

from tqdm import tqdm

from torchvision.transforms import Compose
from transforms.scene import SeqToTensor, Get_cat_shift_info, Padding_joint

from datasets.suncg_shift_seperate_dataset_deepsynth import SUNCG_Dataset
from utils.config import read_config
from utils.room import get_corners


def run_training(cfg):
    t = Compose([Get_cat_shift_info(cfg), Padding_joint(cfg), SeqToTensor()])

    dataset = SUNCG_Dataset(data_folder=cfg["data"]["data_path"], transform=t)

    for sample in tqdm(dataset):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_training(cfg)
