"""
Read the pre-rendered TopDownViews of GT rooms
Create the full RenderedComposite image (37 channel)
Save to file
"""
from pathlib import Path
from glob import glob
from data import RenderedScene
from tqdm import tqdm
import os
import random

import torch

DATA_DIR = "/home/chandan/suncg/bedroom"
OUT_DIR = "/home/chandan/suncg/composite/gt"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gt_files = glob(f"{DATA_DIR}/*.pkl")
    # choose these many files
    n_choice = 1200
    chosen_files = random.sample(gt_files, k=n_choice)
    print(f"Rendering {len(chosen_files)} rooms")

    for path in tqdm(chosen_files):
        index = Path(path).stem
        # the file has to be called n.pkl, n=1,2,3..
        scene = RenderedScene(index, "", DATA_DIR)
        composite = scene.create_composite()

        composite.add_nodes(scene.object_nodes)
        composite_img = composite.get_composite(num_extra_channels=0, ablation="basic")

        out_file = f"{index}.pth"
        torch.save(composite_img, Path(OUT_DIR) / out_file)
    print("Image shape:", composite_img.shape)


if __name__ == "__main__":
    main()
