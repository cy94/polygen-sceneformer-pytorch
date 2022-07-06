from datasets.ply_dataset import PLYDataset
from transforms.polygen import SortMesh, MeshToSeq, QuantizeVertices8Bit, MeshToTensor
from transforms.polygen import get_old_to_new_ndx

from torchvision.transforms import Compose

import pytest

DATA_PATHS = ["tests/data/cube", "tests/data/simple_mesh"]


def test_sort_ndx_map():
    x = [3, 2, 1]
    _, ndx = zip(*sorted((e, i) for (i, e) in enumerate(x)))
    old_to_new = get_old_to_new_ndx(ndx)
    assert old_to_new == {0: 2, 1: 1, 2: 0}


@pytest.mark.parametrize("data_path", DATA_PATHS)
def test_ply_load(data_path):
    dataset = PLYDataset(data_path)

    for sample in dataset:
        vertices, faces = sample["vertices"], sample["faces"]

        assert len(vertices) >= 8
        assert len(faces) >= 6

        for face in faces:
            assert len(face) >= 3


@pytest.mark.parametrize("data_path", DATA_PATHS)
def test_sort_transform(data_path):
    dataset = PLYDataset(data_path, transform=SortMesh())

    for sample in dataset:
        vertices, faces = sample["vertices"], sample["faces"]

        for (i, vertex) in enumerate(vertices[:-1]):
            # z coordinate
            assert vertices[i][2] <= vertices[i + 1][2]

        for face in faces:
            assert face[-1] > face[0]


@pytest.mark.parametrize("data_path", DATA_PATHS)
def test_quantize_vertices(data_path):
    dataset = PLYDataset(
        data_path, transform=QuantizeVertices8Bit(remove_duplicates=True)
    )

    possible_values = range(0, 256)

    for sample in dataset:
        vertices, faces = sample["vertices"], sample["faces"]

        for vtx in vertices:
            for n in vtx:
                assert n in possible_values


@pytest.mark.parametrize("data_path", DATA_PATHS)
def test_seq_transform(data_path):
    dataset = PLYDataset(data_path, transform=MeshToSeq())

    for sample in dataset:
        vertices, faces = sample["vertices"], sample["faces"]
        # vertices has 1 dimension
        assert len(vertices.shape) == 1


def test_vtx_seq_transform():
    t = transform = MeshToSeq(
        vtx_start_token=128, vtx_stop_token=129, vtx_pad_token=130, vtx_seq_len=32
    )
    dataset = PLYDataset("tests/data/cube", transform=t)

    for sample in dataset:
        vertices, faces = sample["vertices"], sample["faces"]

        assert vertices[0] == 128
        assert vertices[-1] == 130
        assert 128 in vertices


@pytest.mark.parametrize("data_path", DATA_PATHS)
def test_stop_token(data_path):
    t = MeshToSeq(vtx_stop_token=128, face_stop_token=-1)
    dataset = PLYDataset(data_path, transform=t)

    for sample in dataset:
        vertices, faces = sample["vertices"], sample["faces"]
        # last token is a stop token
        assert vertices[-1] == 128
        assert faces[-1] == -1


@pytest.mark.parametrize("data_path", DATA_PATHS)
def test_tensor_transform(data_path):
    t = Compose([MeshToSeq(), MeshToTensor()])
    dataset = PLYDataset(data_path, transform=t)

    for sample in dataset:
        vertices, faces = sample["vertices"], sample["faces"]


@pytest.mark.parametrize("data_path", DATA_PATHS)
def test_full_transform(data_path):
    t = Compose(
        [
            QuantizeVertices8Bit(remove_duplicates=True),
            SortMesh(),
            MeshToSeq(vtx_stop_token=128, face_stop_token=-1),
            MeshToTensor(),
        ]
    )
    dataset = PLYDataset(data_path, transform=t)

    for sample in dataset:
        vertices, faces = sample["vertices"], sample["faces"]
