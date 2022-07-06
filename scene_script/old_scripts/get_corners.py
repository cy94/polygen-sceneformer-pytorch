import numpy as np
import cv2
import pickle
from data.dataset import toimage

import torch

from models.resnet import resnet_small


def main():
    model = resnet_small()

    pkl_path = "/shared/data/bedroom/63.pkl"

    with open(pkl_path, "rb") as f:
        data, _ = pickle.load(f)
    floor, _, _ = data

    floor = floor.numpy()
    print(floor.shape, floor.min(), floor.max())
    floor[floor >= 0.1] = 255
    floor[floor < 0.1] = 0

    out = model(torch.Tensor(floor).view(1, 1, 512, 512))
    print("out", out.shape)

    return
    thresh = floor.astype(np.uint8)

    print(thresh.shape, thresh.min(), thresh.max())

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    print(len(contours[0]))

    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for point in contours[0]:
        (x, y) = point[0]
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    cv2.imwrite("/home/chandan/out/63.jpg", img)


if __name__ == "__main__":
    main()
