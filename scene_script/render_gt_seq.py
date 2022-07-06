"""
Render GT scenes from their sequences
"""
import argparse
from tqdm import tqdm

from datasets.suncg_shift_seperate_dataset_deepsynth import SUNCG_Dataset

from torchvision.transforms import Compose
from transforms.scene import SeqToTensor, Get_cat_shift_info, AddStartToken

from utils.config import read_config
from scene_script.test_separate_models_lt import scene_gen


def render(cfg):
    print("Rendering GT seqs")
    t = Compose([Get_cat_shift_info(), AddStartToken(cfg), SeqToTensor()])
    dataset = SUNCG_Dataset(
        data_folder=cfg["data"]["data_path"],
        list_path=cfg["data"]["list_path"],
        transform=t,
    )

    for ndx, scene in enumerate(tqdm(dataset)):
        cat, x, y, z, ori, modelids = (
            scene["cat_seq"],
            scene["x_loc_seq"],
            scene["y_loc_seq"],
            scene["z_loc_seq"],
            scene["orient_seq"],
            scene["modelids"],
        )
        scene_gen(cat, x, y, z, ori, ndx=ndx, out_dir="gt", modelids=modelids)
        if ndx == 1199:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    render(cfg)
