import json
import os, os.path as osp
import pickle
from tqdm import tqdm

# class House():
#     def __init__(self, index=0):
#         self.data_folder = "/shared/data/suncg/house/"
#         houses = dict(enumerate(os.listdir(self.data_folder)))
#         self.__dict__ = json.loads(open(self.data_folder + houses[index] + "/house.json", 'r').read())

data = []

data_folder = "/shared/data/suncg/house/"
all_house_folder = os.listdir(data_folder)
batch_size = 1
cur_idx = 0
for house_folder in tqdm(all_house_folder):
    path = osp.join(data_folder, house_folder, "house.json")
    f = open(path)
    house = json.load(f)

    for level in house["levels"]:
        nodes = level["nodes"]
        for node in nodes:
            if (
                "roomTypes"
                and "nodeIndices" in node.keys()
                and node["roomTypes"] == ["Bedroom"]
            ):
                dic = {}
                objects = []
                dic["room"] = node

                for idx in node["nodeIndices"]:
                    objects.append(nodes[idx])
                dic["objects"] = objects
                dic["houseId"] = house["id"]
                dic["levelId"] = level["id"]
                # data.append(dic)

                # if len(data) == batch_size:
                with open(f"datasets/suncg_bedroom/Bedroom_{cur_idx}.pkl", "wb") as f:
                    pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

                cur_idx += 1
