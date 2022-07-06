import argparse
import sys
import glob
import random
from pathlib import Path

import bpy

import os, os.path as osp

# clear the scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# 20 degrees in radian
max_angle = 0.349066

# taken from
# https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def decimate(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    n_files = len(os.listdir(input_path))
    for ndx, fname in enumerate(os.listdir(input_path)):
        print(f"{ndx} / {n_files}")

        mesh_path = osp.join(input_path, fname)
        bpy.ops.import_mesh.ply(filepath=mesh_path)

        # add modifier
        bpy.ops.object.modifier_add(type="DECIMATE")

        # sample an angle between 0 and max_angle
        angle = random.random() * max_angle

        bpy.context.object.modifiers["Decimate"].angle_limit = angle
        bpy.context.object.modifiers["Decimate"].decimate_type = "DISSOLVE"
        bpy.ops.object.modifier_apply(apply_as="DATA", modifier="Decimate")

        # export mesh with the same filename
        out_fname = fname
        out_path = osp.join(output_path, out_fname)
        bpy.ops.export_mesh.ply(
            filepath=out_path, use_normals=False, use_uv_coords=False, use_colors=False
        )

        # clear mesh
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()


if __name__ == "__main__":
    parser = ArgumentParserForBlender(
        description="Read PLY from file, augment and write back to file"
    )
    parser.add_argument("input_path", help="Path to input PLY files")
    parser.add_argument("output_path", help="Path to input PLY files")

    args = parser.parse_args()

    decimate(args.input_path, args.output_path)
