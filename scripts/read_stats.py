"""
read dataset stats from .npy file, compute more stats and get 
histogram
"""

from itertools import chain

import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch


def compute_stats(fname):
    stats = torch.load(fname)

    plt.hist(stats["vtx_per_mesh"], bins=50)
    plt.xlabel("Vertices per mesh")
    plt.ylabel("Num meshes")
    plt.savefig("vtx_per_mesh.png")
    plt.close()

    plt.hist(stats["face_per_mesh"], bins=50)
    plt.xlabel("Faces per mesh")
    plt.ylabel("Num meshes")
    plt.savefig("face_per_mesh.png")
    plt.close()

    vtx_per_face_flat = list(chain.from_iterable(stats["vtx_per_face"]))
    plt.hist(vtx_per_face_flat, bins=50)
    plt.xlabel("Vertices per face")
    plt.ylabel("Num faces")
    plt.savefig("vtx_per_face.png")
    plt.close()

    max_vtx = max(stats["vtx_per_mesh"])
    print(f"Vertex: Max vertices in a mesh (max_vertices): {max_vtx}")
    print(f"Vertex: max_seq_len: \t\t\t\t{max_vtx * 3 + 2}")

    max_vtx_per_face = max(chain.from_iterable(stats["vtx_per_face"]))
    face_seq_lens = [sum(counts) + len(counts) for counts in stats["vtx_per_face"]]
    print("zeros", np.where(np.array(face_seq_lens) == 0))
    max_face_seq = max(face_seq_lens)
    max_face_per_mesh = max(stats["face_per_mesh"])
    print(f"Face: Max faces in a mesh (max_face_num): {max_face_per_mesh}")
    print(f"Face: Max vertices in a face (max_vtx_per_face): {max_vtx_per_face}")
    print(f"Face: max_seq_len: {max_face_seq + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", help="Path to npy file")

    args = parser.parse_args()

    compute_stats(args.npy_path)
