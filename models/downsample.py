import mesh_simplifier
import numpy as np


def downsample_mesh(vert_pos, faces, ratio=0.5):
    n_target = int(faces.shape[0] * ratio)
    vert_pos = vert_pos.astype(np.float64)
    faces = faces.astype(np.int32)
    flag, V, F, seq = mesh_simplifier.create_decimate_sequence(vert_pos.copy(), faces.copy(), n_target)
    return V.astype(np.float32), F, flag


def downsample_by_sequence(vert_pos, faces, seq):
    vert_pos = vert_pos.astype(np.float64)
    faces = faces.astype(np.int32)
    V, F = mesh_simplifier.decimate_by_sequence(vert_pos, faces, seq)
    return V.astype(np.float32), F


def create_downsample_sequence(vert_pos, faces, ratio=0.5, n_target=None):
    if n_target is None:
        n_target = int(faces.shape[0] * ratio)
    vert_pos = vert_pos.astype(np.float64)
    faces = faces.astype(np.int32)
    _, _, _, seq = mesh_simplifier.create_decimate_sequence(vert_pos, faces, n_target)
    return seq
