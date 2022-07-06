import json
import csv
import numpy as np
import transforms3d
from scipy.spatial.transform import Rotation as R
import pandas as pd
import trimesh
import os
import os.path as osp
import pickle


room_type = 'living'
scene_id = '29_0_6_final_6'
scene_path = f'/home/xinpeng/git/deep-synth/deep-synth/synth_out_{room_type}/{scene_id}.json'


model_to_categories = {}
model_cat_file = "datasets/ModelCategoryMapping.csv"
room_size_cap = [6.05, 4.05, 6.05]
with open(model_cat_file, "r") as f:
    categories = csv.reader(f)
    for l in categories:
        model_to_categories[l[1]] = [l[2], l[3], l[5]]
with open(f'/home/xinpeng/git/deep-synth/deep-synth/data/{room_type}/final_categories_frequency', 'r') as f:
    cat_freq = {}
    lines = f.readlines()
    for line in lines:
        cat_freq[line.split()[0]] = line.split()[1]

def get_final_category(model_id):
    """
    Final categories used in the generated dataset
    Minor tweaks from fine categories
    """
    model_id = model_id.replace("_mirror","")
    category = model_to_categories[model_id][0]
    if model_id == "199":
        category = "dressing_table_with_stool"
    if category == "nightstand":
        category = "stand"
    if category == "bookshelf":
        category = "shelving"
    return category
def scale(loc, orient=None, dim=None):
    loc *= 200/(np.array(room_size_cap) * 1.5)
    if orient is not None:
        orient += 180
        #orient *= 100
    if dim is not None:
        dim *= 200
    return loc, orient #, dim
def add_wall(scene, house_id, room_id, bbox):

    # read wall mesh
    path_root = f'/shared/data/suncg/room/{house_id}/'
    floor_id = f'{room_id}f.obj'
    wall_id = f'{room_id}w.obj'

    T = - (np.array(bbox['min']) * 0.5 + np.array(bbox['max']) * 0.5 - np.array(room_size_cap) * 1.5 * 0.5)

    orient = 0
    r = R.from_euler('y', orient, degrees=True)  # .cpu(), degrees=True)

    Rotation = r.as_matrix()
    Z = [1, 1, 1]
    transform = transforms3d.affines.compose(T, Rotation, Z, S=None)

    floor_file = osp.join(path_root, floor_id)
    floor_mesh = trimesh.load_mesh(floor_file)
    wall_file = osp.join(path_root, wall_id)
    wall_mesh = trimesh.load_mesh(wall_file)

    # transform the wall
    wall_mesh.apply_transform(transform)
    scene.add_geometry(wall_mesh)

    # checker.add_object(name=ndx, mesh=mesh_trimesh)
    # for i in scene.geometry.items():
    #     bbox_dic[i[0]] = i[1].bounds




    floor_mesh.apply_transform(transform)
    scene.add_geometry(floor_mesh)
    # scene.add


    return scene
def scene_gen(cat_gen_seq, x_gen_seq, y_gen_seq, z_gen_seq, orient_gen_seq, modelId_seq, house_id, room_id, room_bbox, ndx):
    with open('datasets/final_categories_frequency', 'r') as f:
        cat_freq = json.load(f)
    cat_list = list(cat_freq.keys())
    model_cat_file = "datasets/ModelCategoryMapping.csv"
    df = pd.read_csv(model_cat_file)

    obj_path_root = '/shared/data/suncg/object'  # '/Volumes/Elements/object'#

    cat = np.array(cat_gen_seq)
    x = np.array(x_gen_seq) * 6.05 * 1.5 / 200
    y = np.array(y_gen_seq) * 4.05 * 1.5 / 200
    z = np.array(z_gen_seq) * 6.05 * 1.5 / 200
    orient = np.array(orient_gen_seq) - 180
    count = 0
    for i in range(len(cat)):
        if cat[i] < len(cat_freq):

            T = np.array([x[i], y[i], z[i]])

            r = R.from_euler('y', orient[i], degrees=True)  # .cpu(), degrees=True)

            Rotation = r.as_matrix()
            Z = [1, 1, 1]
            transform = transforms3d.affines.compose(T, Rotation, Z, S=None)
            model_cat = cat_list[cat[i]]
            # model_id = df[df['fine_grained_class'] == model_cat][:1]['model_id'].item()
            model_id = modelId_seq[i]
            file = f'{model_id}/{model_id}.obj'
            obj_path = osp.join(obj_path_root, file)
            mesh = trimesh.load_mesh(obj_path)
            mesh.apply_transform(transform)
            if count == 0:
                scene = trimesh.scene.scene.Scene(geometry=mesh)
            scene.add_geometry(mesh)
            count += 1

    # add wall and flore to the scene
    scene = add_wall(scene, house_id, room_id, room_bbox)


    out_path = '/home/xinpeng/dlcv-ss20/deep_synth/mesh'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    scene_file = osp.join(out_path, f'{ndx}.glb')
    scene.export(scene_file)
def get_seq_from_scene_info(scene_path):
    with open(scene_path, 'r') as f:
        scene_info = json.load(f)
    cat_seq = np.array([])
    x_loc_seq = np.array([])
    y_loc_seq = np.array([])
    z_loc_seq = np.array([])
    orient_seq = np.array([])
    modelId_seq = np.array([])

    room_type = 'bedroom'
    objects = scene_info['levels'][0]['nodes']
    # objects_sorted =sort_filter_obj(objects, room_type)
    room_bbox = objects[0]['bbox']
    for obj in objects[1:]:
        cat = get_final_category(obj['modelId'])
        cat_idx = np.where(np.array(list(cat_freq.keys())) == cat)[0]
        trans = np.array(obj['transform']).reshape(4, 4).T
        shift = - (np.array(room_bbox['min']) * 0.5 + np.array(room_bbox['max']) * 0.5 - np.array(
            room_size_cap) * 1.5 * 0.5)

        loc = transforms3d.affines.decompose44(trans)[0] + shift
        rot_matrix = transforms3d.affines.decompose44(trans)[1]
        r = R.from_matrix(rot_matrix)
        orient = r.as_euler('yzx', degrees=True)[0]
        # scale
        loc, orient = scale(loc=loc, orient=orient)

        # modelId
        modelId = obj['modelId']
        x, y, z = loc[0], loc[1], loc[2]
        x_loc_seq = np.hstack((x_loc_seq, x))
        y_loc_seq = np.hstack((y_loc_seq, y))
        z_loc_seq = np.hstack((z_loc_seq, z))
        cat_seq = np.hstack((cat_seq, cat_idx))
        orient_seq = np.hstack((orient_seq, orient))
        modelId_seq  = np.hstack((modelId_seq, modelId))

    cat_seq = [int(i) for i in cat_seq]
    room_id = objects[0]['modelId']
    house_id = scene_info['id']



    return cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq, modelId_seq, house_id,  room_id, room_bbox
def store_deepsynth_scenes():
    deep_out_folder = '/home/xinpeng/git/deep-synth/deep-synth/synth_out'
    all_files = os.listdir(deep_out_folder)

    json_files = list(filter(lambda s: s.endswith('.json'), all_files))

    final = list(filter(lambda fname: 'final' in fname, json_files))
    scene_folder = '/home/xinpeng/git/deep-synth/deep-synth/synth_out'

    out_path = 'deep_synth/pickle'

    for file in final:
        deep = {}
        scene_path = osp.join(scene_folder,file)
        cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq, modelId_seq = get_seq_from_scene_info(scene_path)
        deep['cat_seq'] = cat_seq
        pickle_file = osp.join(out_path, f'{file[:-5]}.pkl')
        with open(pickle_file, "wb") as f:
            pickle.dump(deep, f, pickle.HIGHEST_PROTOCOL)

# get info from synthed scene
cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq, modelId_seq,  house_id, room_id, room_bbox = get_seq_from_scene_info(scene_path)
# generate scene mesh
scene_gen(cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq, modelId_seq,house_id, room_id,room_bbox,scene_id)




# store_deepsynth_scenes()

# get_info(scene_info)
#
#     return seq
# scene_gen(seq)
