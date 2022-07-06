"""
read dataset stats from .pth file, compute more stats and get 
histogram

usage: python scripts/plot_stats.py --file <file1> <title1> --file <file2> <title2> ...
"""

from itertools import chain

import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch


def compute_stats(files):
    all_stats = []

    for fname, title in files:
        print(f"Loading file: {fname}, title: {title}")
        stats = torch.load(fname)
        all_stats.append((title, stats))

    labels = [e[0] for e in all_stats]

    vertices = [e[1]["vtx_per_mesh"] for e in all_stats]
    plt.hist(vertices, bins=50, histtype="step", density=True, label=labels)
    plt.xlabel("Vertices per mesh")
    plt.ylabel("Num meshes")
    plt.legend()
    plt.savefig("vtx_per_mesh.png")
    plt.close()

    faces = [e[1]["face_per_mesh"] for e in all_stats]
    plt.hist(faces, bins=50, density=True, histtype="step", label=labels)
    plt.xlabel("Faces per mesh")
    plt.ylabel("Num meshes")
    plt.legend()
    plt.savefig("face_per_mesh.png")
    plt.close()

    vtx_per_face = []
    for e in all_stats:
        vtx_per_face.append(list(chain.from_iterable(e[1]["vtx_per_face"])))

    plt.hist(vtx_per_face, bins=50, density=True, histtype="step", label=labels)
    plt.xlabel("Vertices per face")
    plt.ylabel("Num faces")
    plt.legend()
    plt.savefig("vtx_per_face.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action="append", nargs=2)

    args = parser.parse_args()

    compute_stats(args.file)
