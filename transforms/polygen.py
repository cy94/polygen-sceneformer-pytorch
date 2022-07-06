import torch
import numpy as np
import copy


def get_old_to_new_ndx(sorted_ndx):
    """
    sorted_ndx: the permutation index list
    eg: sorted((3,2,1)) -> sorted list = (1,2,3) and sorted_ndx=(2,1,0)

    output: dict mapping from old index of an element to its new index
    eg: {0:2, 1:1, 2:0} (index=0 element is now at index=2, 
                         index=1 element is now at index=1,
                         index=2 element is now at index=0) 
    """
    mapping = {}
    for new_pos, old_pos in enumerate(sorted_ndx):
        mapping[old_pos] = new_pos
    return mapping


def update_faces(faces, vtx_old_to_new):
    """
    faces: list of faces (list of list of vertex indices)
    vtx_old_to_new: mapping from old to new vertex indices
    """
    new_faces = []

    for face in faces:
        new_face = list(map(lambda old_vtx: vtx_old_to_new[old_vtx], face))
        new_faces.append(new_face)

    return new_faces


class SortMesh(object):
    """
    Convert an unordered lists of vertices and faces to sorted lists
    as done in the PolyGen paper
    """

    def __call__(self, sample):
        # make a copy and modify the copy, dont change the original sample
        sample = copy.deepcopy(sample)
        vertices, faces = sample["vertices"], sample["faces"]

        # get (vertex, index) pairs
        vtx_ndx = ((vtx, i) for (i, vtx) in enumerate(vertices))
        # sorts by vertex.z, then vertex.y, then vertex.x
        sorted_vtx_ndx = sorted(vtx_ndx, key=lambda vi: (vi[0][2], vi[0][1], vi[0][0]))
        # then back to np array
        sorted_vertices, sorted_ndx = zip(*sorted_vtx_ndx)
        sorted_vertices = np.array([np.array(v) for v in sorted_vertices])

        # mapping from old to new vertices
        old_to_new = get_old_to_new_ndx(sorted_ndx)

        # update vertex indices in all faces
        new_faces = update_faces(faces, old_to_new)

        # sort the faces by lowest index vertex, next lowest index vertex ..
        # sort within the face to get the lowest, next lowest vertex
        sorted_faces = np.array(sorted(new_faces, key=lambda face: sorted(face)))

        # in each face, make the lowest-index vertex the first
        # by permuting circularly
        for (ndx, face) in enumerate(sorted_faces):
            lowest_vtx_pos = np.argmin(face)
            sorted_faces[ndx] = np.roll(face, -lowest_vtx_pos)

        # update sample
        sample["vertices"] = sorted_vertices
        sample["faces"] = sorted_faces

        return sample


class QuantizeVertices8Bit(object):
    """
    Quantize x,y,z values into bins: (0, 1, 2 .. 255)
    """

    def __init__(self, remove_duplicates=False, scale=16):
        """
        remove_duplicates: if 2 vertices are the same after quantizing
                            one is removed
        scale: maximum positive vertex coordinate after scaling                            
        """
        self.remove_duplicates = remove_duplicates
        self.scale = scale

    def __call__(self, sample):
        # make a copy and modify the copy, dont change the original sample
        sample = copy.deepcopy(sample)

        vertices, faces = sample["vertices"], sample["faces"]

        # make all coordinates positive
        vertices -= vertices.min(axis=0)
        max_coordinate = vertices.max()
        # scale everything to a maximum value
        vertices = (vertices * self.scale) / max_coordinate
        # round to nearest integer
        vertices = np.round_(vertices)
        # convert to integer type
        # everything should be in 0-255 by now
        assert vertices.min() >= 0 and vertices.max() <= 255
        quant_vertices = vertices.astype(np.uint8)

        if self.remove_duplicates:
            # remove duplicate vertices
            quant_vertices, ndx = np.unique(quant_vertices, axis=0, return_inverse=True)
            # mapping from old vertex indices to new vertex indices
            mapping_old_to_new = dict(zip(range(len(vertices)), ndx))
            faces = update_faces(faces, mapping_old_to_new)

        # vertex indices may repeat within a face
        # this can make a face degenerate (line, point)
        # keep only faces in which all vertices are unique
        faces = list(filter(lambda face: len(np.unique(face)) == len(face), faces))

        sample["vertices"] = quant_vertices
        sample["faces"] = faces

        return sample


class GetFacePositions(object):
    """
    Get the face position index and the in-face position of the vertices
    """

    def __init__(self, new_face_token=None):
        self.new_face_token = new_face_token

    def __call__(self, sample):
        # make a copy and modify the copy, dont change the original sample
        sample = copy.deepcopy(sample)

        faces = sample["faces"]
        vertices = sample["vertices"]
        # pos : 00N111N222N33N444
        pos_ndx = faces.copy()
        for (ndx, face) in enumerate(pos_ndx):
            pos_ndx[ndx] = np.ones(len(pos_ndx[ndx])) * (ndx)
        # add new_face_token
        new_pos = pos_ndx.tolist()
        for (ndx, face) in enumerate(pos_ndx):
            if ndx <= len(pos_ndx) - 2:
                face_with_stop = np.hstack((face, [self.new_face_token]))
                new_pos[ndx] = face_with_stop
        # flatten
        flat_new_pos = np.hstack(new_pos)

        # todo: inface: 1,2,3,N,1,2,3,4,N
        in_face = faces.copy()
        for (ndx, face) in enumerate(in_face):
            in_face[ndx] = range(len(in_face[ndx]))
        new_inface = in_face.tolist()
        for (ndx, face) in enumerate(in_face):
            if ndx <= len(in_face) - 2:
                face_with_stop = np.hstack((face, [self.new_face_token]))
                new_inface[ndx] = face_with_stop
        # flatten
        flat_new_inface = np.hstack(new_inface)

        sample["face_pos"] = flat_new_pos
        sample["in_face_pos"] = flat_new_inface

        return sample


class RandomScale(object):
    def __init__(self, low=0.75, high=1.25):
        self.low = low
        self.high = high

    def __call__(self, sample):
        # make a copy and modify the copy, dont change the original sample
        sample = copy.deepcopy(sample)

        # select scale factors for x, y, z
        scale = np.random.uniform(self.low, self.high, 3)

        sample["vertices"] *= scale

        return sample


class YUpToZUp(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # make a copy and modify the copy, dont change the original sample
        sample = copy.deepcopy(sample)

        vertices = sample["vertices"]
        sample["vertices"] = vertices[:, (0, 2, 1)]

        return sample


class RandomLinearWarp(object):
    """
    Apply random linear warp to x,y,z coordinates
    The warp is symmetric about the x and y axes
    (See PolyGen paper for more details)
    
    Divide the values along each axis into num_pieces
    Pick a warp value from a lognormal distribution with variance 0.5
    and multiply by the warp value
    """

    def __init__(self, num_pieces=6, warp_var=0.5):
        assert num_pieces % 2 == 0, f"num_pieces should be even, got {num_pieces}"

        self.num_pieces = num_pieces
        self.warp_var = warp_var

    def get_warp(self, coord_ndx):
        """
        coord_ndx: 0 -> x, 1 -> y, 2 -> z
        """
        if coord_ndx in (0, 1):
            w = np.random.lognormal(0, self.warp_var, self.num_pieces // 2)
            warp = np.hstack([w, np.flip(w)])
        elif coord_ndx == 2:
            warp = np.random.lognormal(0, self.warp_var, self.num_pieces)
        else:
            raise ValueError
        return warp

    def __call__(self, sample):
        # make a copy and modify the copy, dont change the original sample
        sample = copy.deepcopy(sample)

        vertices = sample["vertices"]
        min_vals, max_vals = vertices.min(axis=0), vertices.max(axis=0)
        # x, y, z value ranges
        coord_ranges = list(zip(min_vals, max_vals))
        # loop over x, y, z
        for coord_ndx, coord_range in enumerate(coord_ranges):
            # coordinate interval endpoints
            endpoints = np.linspace(coord_range[0], coord_range[1], self.num_pieces + 1)
            # coordinate intervals
            intervals = list(zip(endpoints[:-1], endpoints[1:]))
            # warp value for each interval
            warp = self.get_warp(coord_ndx)
            # apply warp to each interval
            for ndx, (low, high) in enumerate(intervals):
                # select only X, Y or Z
                coords = vertices[:, coord_ndx]
                # transform values in this interval and replace
                coords[(coords > low) & (coords < high)] *= warp[ndx]
                # replace back into original vertices set
                vertices[:, coord_ndx] = coords

        sample["vertices"] = vertices

        return sample


class FilterMesh(object):
    """
    Return None for meshes that have more than the specified number of vertices 
    or faces
    This is handled later by nonechucks, which discards the sample
    """

    def __init__(self, vtx_limit=800, face_limit=600):
        self.vtx_limit = vtx_limit
        self.face_limit = face_limit

    def __call__(self, sample):
        vertices, faces = sample["vertices"], sample["faces"]
        if len(vertices) < 3 or len(faces) < 3:
            return None
        if len(vertices) > self.vtx_limit or len(faces) > self.face_limit:
            return None

        return sample


class MeshToSeq(object):
    """
    Flatten a mesh with (vertices, faces) into a sequence of tokens

    Vertices: (x1,y1,z1), (x2,y2,z2) .. -> (z1, y1, x1, z2, y2, x2, ..)
    Faces: 
    """

    def __init__(
        self,
        vtx_start_token=None,
        vtx_stop_token=None,
        vtx_pad_token=None,
        vtx_seq_len=None,
        new_face_token=None,
        face_pad_token=None,
        face_seq_len=None,
        face_stop_token=None,
    ):
        self.vtx_start_token = vtx_start_token
        self.vtx_stop_token = vtx_stop_token
        self.vtx_pad_token = vtx_pad_token
        self.vtx_seq_len = vtx_seq_len

        self.new_face_token = new_face_token
        self.face_pad_token = face_pad_token
        self.face_seq_len = face_seq_len
        self.face_stop_token = face_stop_token

    def __call__(self, sample):
        # make a copy and modify the copy, dont change the original sample
        sample = copy.deepcopy(sample)

        vertices, faces = sample["vertices"], sample["faces"]
        flat_new_pos, flat_new_inface = sample["face_pos"], sample["in_face_pos"]
        # reorder the columns xyz->zyx, then flatten
        flat_vertices = np.hstack(vertices[:, ::-1])

        # add a single start token
        if self.vtx_start_token:
            flat_vertices = np.hstack(([self.vtx_start_token], flat_vertices))
        # add a single stop token
        if self.vtx_stop_token:
            flat_vertices = np.hstack((flat_vertices, [self.vtx_stop_token]))

        if self.vtx_pad_token:
            n_pad = self.vtx_seq_len - len(flat_vertices)
            assert n_pad >= 0
            flat_vertices = np.pad(
                flat_vertices,
                (0, n_pad),
                mode="constant",
                constant_values=self.vtx_pad_token,
            )
        # add stop token at the end of each face
        new_faces = faces.tolist()

        if self.new_face_token:
            # dont add a new face token after the last face
            for (ndx, face) in enumerate(new_faces):
                if ndx <= len(new_faces) - 2:
                    face_with_new = np.hstack((face, [self.new_face_token]))
                    new_faces[ndx] = face_with_new
            # flatten only if there is a stop token
            flat_faces = np.hstack(new_faces)

        if self.face_stop_token:
            flat_faces = np.hstack((flat_faces, [self.face_stop_token]))
            flat_new_inface = np.hstack((flat_new_inface, [self.face_stop_token]))
            flat_new_pos = np.hstack((flat_new_pos, [self.face_stop_token]))

        if self.face_pad_token:
            n_pad = self.face_seq_len - len(flat_faces)
            assert n_pad >= 0
            flat_faces = np.pad(
                flat_faces,
                (0, n_pad),
                mode="constant",
                constant_values=self.face_pad_token,
            )

            flat_new_pos = np.pad(
                flat_new_pos,
                (0, n_pad),
                mode="constant",
                constant_values=self.face_pad_token,
            )

            flat_new_inface = np.pad(
                flat_new_inface,
                (0, n_pad),
                mode="constant",
                constant_values=self.face_pad_token,
            )

        sample["vertices_raw"] = vertices
        sample["faces_raw"] = faces
        sample["vertices"] = flat_vertices
        sample["faces"] = flat_faces if self.new_face_token else faces
        sample["face_pos"] = flat_new_pos
        sample["in_face_pos"] = flat_new_inface

        return sample


class MeshToTensor(object):
    """
    Convert vertices and faces to tensors
    """

    def __call__(self, sample):
        # make a copy and modify the copy, dont change the original sample
        sample = copy.deepcopy(sample)

        convert_list = "vertices", "faces", "face_pos", "in_face_pos"

        return {
            k: torch.LongTensor(sample[k]) if k in convert_list else sample[k]
            for k in sample
        }
