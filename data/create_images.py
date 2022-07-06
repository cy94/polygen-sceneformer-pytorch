import torch


def create_wall(size=512, fillval=0.711):
    wall = torch.zeros((size, size))
    # top
    wall[:8, :] = fillval
    # bottom
    wall[-8:, :] = fillval
    # left
    wall[:, :8] = fillval
    # right
    wall[:, -8:] = fillval

    return wall


def create_floor(size=512, fillval=0.1209):
    floor = torch.ones((size, size)) * fillval
    return floor
