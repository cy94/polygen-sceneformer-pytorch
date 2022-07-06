import argparse
from pathlib import Path
import os, os.path as osp

from tqdm import tqdm

import pymesh


def augment(args):
    """
    args: full args from command line
    """
    os.makedirs(args.output_path, exist_ok=True)
    filtered = 0
    for object_dir in tqdm(os.listdir(args.input_path)):
        obj_path = osp.join(args.input_path, object_dir, "model.obj")
        mesh = pymesh.load_mesh(obj_path)

        if len(mesh.vertices) < args.filter_vtx and len(mesh.faces) < args.filter_face:
            filtered += 1
            out_file = f"{object_dir}.ply"
            out_path = osp.join(args.output_path, out_file)

            pymesh.save_mesh(out_path, mesh)
    print(f"Saved: {filtered}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read PLY from file, augment and write back to file"
    )
    parser.add_argument(
        "input_path",
        help="Path to input ShapeNet directory (/shared/data/ShapeNetCore.v1/04554684)",
    )
    parser.add_argument("output_path", help="Path to output PLY files directory")
    parser.add_argument("filter_vtx", help="Max vertices in a mesh", type=int)
    parser.add_argument("filter_face", help="Max faces in a mesh", type=int)

    args = parser.parse_args()

    augment(args)
