"""
Calculate KL divergence with GT category distributions and
predicted category distribution
"""
import argparse
from scipy.stats import entropy
import numpy as np
from glob import glob
import torch
import torch.nn.functional as F

NUM_CATS = 31


def get_dist(dir):
    """
    read sequences from dir and calculate the distribution of categories
    """
    counts = np.zeros(NUM_CATS, dtype=np.int64)
    for path in glob(f"{dir}/*.pth"):
        seqs = torch.load(path)
        catseq = seqs["cat"]
        # count categories in each scene
        # get arrays of different lengths
        counts += np.bincount(np.array(catseq), minlength=NUM_CATS)

    distrib = counts / counts.sum()

    return distrib


def get_kl_div(gt_dir, preds_dir):
    gt_dist = get_dist(gt_dir)
    pred_dist = get_dist(preds_dir)
    # KL(synth | dataset) as done in fast and flexible paper
    print("KL Div: ", entropy(pred_dist, gt_dist))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_dir", help="Dir containing GT sequences")
    parser.add_argument("preds_dir", help="Dir containing predicted sequences")
    args = parser.parse_args()

    get_kl_div(args.gt_dir, args.preds_dir)
