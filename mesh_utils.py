import torch
import torch.nn as nn
import igl
import numpy as np
import scipy.sparse as sp
import pickle
from utils import batch_mm, numpy_wrapper, signed_distance, per_face_normals
import sksparse.cholmod as cholmod

import os


def csr2torch(csr):
    coo = csr.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def solver_exception_wrapper(solver):
    def wrapper(*args, **kwargs):
        try:
            res = solver(*args, **kwargs)
        except Exception as e:
            print(e)
            return None
        if np.any(np.isnan(res)):
            print('nan')
            return None
        return res
    return wrapper


class PoissonSolver(nn.Module):
    def __init__(self, V, F, v_c=None):
        super(PoissonSolver, self).__init__()
        if v_c is None or len(v_c) == 0:
            v_c = [V.shape[0] - 1]
        assert isinstance(v_c, list)
        v_c.sort()

        self.v_c_cpu = np.array(v_c, dtype=np.int64)

        set_v = set(list(range(V.shape[0])))
        set_v_c = set(v_c)
        set_v_f = set_v - set_v_c
        v_f = list(set_v_f)
        v_f.sort()
        v_f = torch.tensor(v_f, dtype=torch.long)
        v_c = torch.tensor(v_c, dtype=torch.long)

        self.register_buffer('v_f', v_f)
        self.register_buffer('v_c', v_c)

        self.V = V
        self.F = F

        G = igl.grad(V, F)
        area = igl.doublearea(V, F)
        D_idx = [[], []]
        D_val = []
        for i in range(len(area)):
            for j in range(3):
                D_idx[0].append(i + j * F.shape[0])
                D_idx[1].append(i + j * F.shape[0])
                D_val.append(area[i] / 2)

        D = sp.csc_matrix((D_val, (D_idx[0], D_idx[1])), shape=(3 * F.shape[0], 3 * F.shape[0]))
        L2 = -igl.cotmatrix(V, F)
        self.L = L2

        self.L = self.L.todense()
        self.L = torch.from_numpy(self.L)

        self.Lf = self.L[v_f, :][:, v_f]

        Lc = self.L[v_f, :][:, v_c]
        self.register_buffer('Lc', Lc)

        L_inv = torch.inverse(self.Lf)
        self.register_buffer('L_inv', L_inv.to_sparse())

        RHS_G = G.T @ D
        RHS_G = csr2torch(RHS_G)
        self.register_buffer('RHS_G', RHS_G)

    def forward(self, M, constraints=None):
        """
        :param M: per face jacobian. F x 3 x 3
        :return:
        """
        if constraints is None:
            constraints = torch.tensor(self.V[self.v_c_cpu], device=self.Lc.device)

        M = M.transpose(-1, -2)
        M = M.permute(1, 0, 2)
        M = M.reshape(-1, 3)
        RHS = torch.sparse.mm(self.RHS_G, M)
        RHS = RHS[self.v_f]
        RHS = RHS - self.Lc @ constraints
        res = torch.empty(self.V.shape[0], 3, device=self.Lc.device)
        res[self.v_f] = torch.sparse.mm(self.L_inv, RHS)
        res[self.v_c] = constraints
        return res


class SymbolicPoissonSolver:
    def __init__(self, V, F, v_c=None, persistent=False):
        V = V.astype(np.float64)
        if v_c is None or len(v_c) == 0:
            v_c = [V.shape[0] - 1]
        assert isinstance(v_c, list)
        v_c.sort()

        set_v = set(list(range(V.shape[0])))
        set_v_c = set(v_c)
        set_v_f = set_v - set_v_c
        v_f = list(set_v_f)
        v_f.sort()

        self.v_f = v_f
        self.v_c = v_c
        self.F = F

        Lf, Lc = self.create_laplacian(V)

        self.persistent = persistent
        if persistent:
            self.symbolic_factor = cholmod.cholesky(Lf)
            self.Lc = Lc
        else:
            self.symbolic_factor = cholmod.analyze(Lf)

    def create_laplacian(self, V):
        L = -igl.cotmatrix(V, self.F)
        Lf = L[self.v_f, :][:, self.v_f]
        Lc = L[self.v_f, :][:, self.v_c]
        return Lf, Lc

    def create_rhs(self, V, M):
        F = self.F
        G = igl.grad(V, F)
        area = igl.doublearea(V, F) / 2
        D_val = np.tile(area[None], (3, 1)).reshape(-1)[:, None]
        return G.T @ (D_val * M)

    @solver_exception_wrapper
    def forward(self, V, M, constraints=None):
        """
        :param M: per face jacobian. F x 3 x 3
        :return:
        """
        if constraints is None:
            constraints = V[self.v_c]

        if isinstance(V, torch.Tensor):
            V = V.detach().cpu().numpy().astype(np.float64)
        if isinstance(M, torch.Tensor):
            M = M.detach().cpu().numpy().astype(np.float64)

        M = M.transpose(2, 0, 1)
        M = M.reshape(-1, 3)

        RHS = self.create_rhs(V, M)

        if self.persistent:
            cholesky_solver = self.symbolic_factor
            Lc = self.Lc
        else:
            Lf, Lc = self.create_laplacian(V)
            # cholesky_solver = self.symbolic_factor.cholesky(Lf)
            cholesky_solver = cholmod.cholesky(Lf)

        RHS = RHS[self.v_f]
        RHS = RHS - Lc @ constraints
        res = np.empty((V.shape[0], 3))
        res[self.v_f] = cholesky_solver.solve_A(RHS)
        res[self.v_c] = constraints
        return res


@numpy_wrapper
def get_local_frames(V, F):
    if F.dtype != torch.int64:
        F = F.to(torch.int64)
    v1 = V[..., F[:, 1], :] - V[..., F[:, 0], :]
    v2 = V[..., F[:, 2], :] - V[..., F[:, 0], :]
    N = torch.linalg.cross(v1, v2)
    assert torch.allclose(torch.linalg.cross(v1, v2)[0], torch.cross(v1[0], v2[0]))
    N = N / (torch.norm(N, dim=-1, keepdim=True) + 1e-8)
    local_frames = torch.stack([v1, v2, N], dim=-1)
    return local_frames


@numpy_wrapper
def get_face_normal(V, F):
    v1 = V[..., F[:, 1], :] - V[..., F[:, 0], :]
    v2 = V[..., F[:, 2], :] - V[..., F[:, 0], :]
    N = torch.linalg.cross(v1, v2)
    N = N / (torch.norm(N, dim=-1, keepdim=True) + 1e-8)
    return N


@numpy_wrapper
def get_dg(V1, V2, F):
    frames1 = get_local_frames(V1, F)
    frames2 = get_local_frames(V2, F)
    # frames2[..., 2] = torch.randn((frames2.shape[0],  frames2.shape[1]))
    # This is an experiment showing my implementation is equivalent to NJF.
    dg = frames2 @ torch.linalg.inv(frames1)
    return dg


class FaceNeighbors:
    def __init__(self, F):
        self.ff, self.ffi = igl.triangle_triangle_adjacency(F)
        self.F = F
        self.ff[self.ff == -1] = F.shape[0]

    def __call__(self, features, padding=0):
        """

        :param features: (batch_size, num_faces, n_channels)
        :param padding:
        :return: features: (batch_size, num_faces, num_neighbors, n_channels)
        """
        paddings = torch.empty((features.shape[0], 1, features.shape[2]), dtype=torch.float32, device=features.device)
        paddings.fill_(padding)
        features = torch.cat([features, paddings], dim=1)

        return torch.stack([features[:, self.ff[:, i], :] for i in range(self.ff.shape[1])], dim=2)


def solve_poisson(prev_pos, jacobians, F, list_vc=None):
    if isinstance(prev_pos, torch.Tensor):
        prev_pos = prev_pos.detach().cpu().numpy()
    solver = PoissonSolver(prev_pos, F, list_vc).to(jacobians.device)
    return solver.forward(jacobians)


def high_order_face_neighbor(F, order, bug_free_neighbor, asdense=False):
    # Order == -2 indicates that we want to use the all triangles as neighbor
    assert bug_free_neighbor
    if order == -2:
        return [[i] + list(range(0, i)) + list(range(i+1, F.shape[0])) for i in range(F.shape[0])]

    f_adj, _ = igl.triangle_triangle_adjacency(F)

    coo = [[], []]
    for i in range(F.shape[0]):
        if bug_free_neighbor:
            coo[0].append(i)
            coo[1].append(i)
        for j in range(f_adj.shape[1]):
            k = f_adj[i, j]
            if k != -1:
                coo[0].append(i)
                coo[1].append(k)

    adj_mat = sp.csr_matrix(([1] * len(coo[0]), coo), shape=(F.shape[0], F.shape[0]), dtype=np.bool)

    res = adj_mat.copy()
    for i in range(order - 1):
        res = res @ adj_mat

    if asdense:
        return res.todense()

    coo = res.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    if bug_free_neighbor:
        neighbors = [[i] for i in range(F.shape[0])]
    else:
        neighbors = [[] for _ in range(F.shape[0])]
    for i, idx in enumerate(values):
        if values[i] > 0:
            if indices[0, i] != indices[1, i]:
                neighbors[indices[0, i]].append(indices[1, i])

    return neighbors


@numpy_wrapper
def normalize_rotation(mat):
    r = nearest_rotation(mat)
    return r.transpose(-2, -1) @ mat


@numpy_wrapper
def nearest_rotation(mat):
    shape = mat.shape
    mat = mat.reshape(-1, 3, 3)
    U, s, V = torch.linalg.svd(mat)
    res = U @ V
    det = torch.linalg.det(res)
    mask = det < 0
    if mask.any():
        U[mask, -1, :] = -U[mask, -1, :]
        res = U @ V
    assert (torch.allclose(torch.linalg.det(res), torch.ones_like(det)))
    return res.reshape(shape)


@numpy_wrapper
def translation_alignment(vert, target):
    assert (len(vert.shape) == 3 or len(vert.shape) == 2) and vert.shape == target.shape
    vert = vert - torch.mean(vert, dim=-2, keepdim=True) + torch.mean(target, dim=-2, keepdim=True)
    return vert


@numpy_wrapper
def mean_vertex_error(vert, target, translate_align=False):
    if vert.shape[1] != target.shape[1]:
        min_ = min(vert.shape[1], target.shape[1])
        vert = vert[:, :min_, :]
        target = target[:, :min_, :]
    assert (len(vert.shape) == 3 or len(vert.shape) == 2) and vert.shape == target.shape
    if translate_align:
        vert = translation_alignment(vert, target)
    return torch.mean(torch.norm(vert - target, dim=-1)) * 100


def body_cloth_collision(cloth_pos, body_pos, F_body, collision_threshold=1e-3, eps=0.):
    if isinstance(cloth_pos, torch.Tensor):
        cloth_pos = cloth_pos.detach().cpu().numpy()
    S, I, C = signed_distance(cloth_pos, body_pos, F_body)
    body_vert_norm = per_face_normals(body_pos, F_body)
    new_vert = cloth_pos.copy()
    collision = S < collision_threshold
    new_vert[collision] = C[collision] + eps * body_vert_norm[I[collision]]
    new_vert = torch.from_numpy(new_vert)
    collision = torch.from_numpy(collision).to(torch.float32)
    return new_vert, collision


def write_obj(filename, vs, faces, tc=None, ftc=None):
    faces = faces + 1
    if ftc is not None:
        ftc = ftc + 1
    with open(filename, 'w') as f:
        for vi, v in enumerate(vs):
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        if tc is not None:
            for vi, v in enumerate(tc):
                f.write('vt %f %f\n' % (v[0], v[1]))
        for i, face in enumerate(faces):
            if ftc is None:
                f.write('f %d %d %d\n' % (face[0], face[1], face[2]))
            else:
                f.write('f %d/%d %d/%d %d/%d\n' % (face[0], ftc[i][0], face[1], ftc[i][1], face[2], ftc[i][2]))


def write_vert_pos_pickle(filename, vert_pos, faces):
    if isinstance(vert_pos, torch.Tensor):
        vert_pos = vert_pos.cpu().numpy()
    mesh_sequence_pickle = [{'vertices': vert_pos[i], 'faces': faces} for i in
                            range(vert_pos.shape[0])]
    with open(filename, 'wb') as f:
        pickle.dump(mesh_sequence_pickle, f)


def create_sdf_info(centroids, body_pos, F_body, outward_vec=False, need_C=False):
    S, _, C = signed_distance(centroids, body_pos, F_body)
    return create_sdf_info_from_SC(S, C, centroids, outward_vec, need_C)


def create_sdf_info_from_SC(S, C, centroids, outward_vec=False, need_C=False):
    vec = C - centroids
    if outward_vec:
        vec = -vec * np.sign(S[:, None])
    vec = vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)
    assert np.isnan(vec).sum() == 0
    if need_C:
        return S, C, vec
    else:
        return np.concatenate([S[..., None], vec], axis=-1)


class PenetrationSolver:
    def __init__(self, rest_pos, F, lambda_all, eps=7e-3):
        self.n_vert = rest_pos.shape[0]
        self.F = F
        self.eps = eps
        self.lambda_all = lambda_all

        LHS, _ = self.construct_constraint(rest_pos)
        self.symbolic_factor = cholmod.cholesky(LHS)

    def construct_constraint(self, verts, body_pos=None, F_body=None):
        if body_pos is not None:
            S, C, n = create_sdf_info(verts, body_pos, F_body, outward_vec=True, need_C=True)
            n[S >= self.eps] = 0
        else:
            C = None
            n = np.ones_like(verts)

        a = np.arange(self.n_vert)[:, None]
        b = np.arange(3) * self.n_vert
        y = (a + b).reshape(-1)
        x = np.tile(a, (1, 3)).reshape(-1)

        verts_row = verts.transpose(1, 0).reshape(-1)
        C_row = C.transpose(1, 0).reshape(-1) if C is not None else None

        N_mat = sp.csc_matrix((n.reshape(-1), (x, y)), shape=(self.n_vert, 3 * self.n_vert))
        LHS1 = N_mat.T @ N_mat
        if C is not None:
            RHS1 = N_mat.T @ (N_mat @ C_row + self.eps)
        else:
            RHS1 = 0.

        L = -igl.cotmatrix(verts, self.F)
        L = sp.bmat([[L, None, None], [None, L, None], [None, None, L]])
        LHS2 = L.T @ L
        RHS2 = L.T @ (L @ verts_row)

        LHS3 = sp.eye(3 * self.n_vert)
        RHS3 = verts_row

        LHS = self.lambda_all[0] * LHS1 + self.lambda_all[1] * LHS2 + self.lambda_all[2] * LHS3
        RHS = self.lambda_all[0] * RHS1 + self.lambda_all[1] * RHS2 + self.lambda_all[2] * RHS3

        return LHS, RHS

    @solver_exception_wrapper
    def forward(self, verts, body_pos, F_body):
        LHS, RHS = self.construct_constraint(verts, body_pos, F_body)
        # cholesky_solver = self.symbolic_factor.cholesky(LHS)
        cholesky_solver = cholmod.cholesky(LHS)
        # cholesky_solver = self.symbolic_factor
        res = cholesky_solver.solve_A(RHS)
        return res.reshape(3, -1).T


def create_face_laplace(faces, use_torch=False):
    TT, _ = igl.triangle_triangle_adjacency(faces)
    row = np.arange(faces.shape[0])[:, None].repeat(3, axis=1).reshape(-1)
    flag = TT > -1
    row_cnt = flag.sum(axis=1)
    col = TT.reshape(-1)
    data = 1. / row_cnt
    data = data[:, None].repeat(3, axis=1).reshape(-1)

    flag = flag.reshape(-1)
    data = data[flag]
    row = row[flag]
    col = col[flag]

    L = sp.csc_matrix((data, (row, col)), shape=(faces.shape[0], faces.shape[0])) - sp.eye(faces.shape[0])

    if use_torch:
        L = csr2torch(L)
    return L


def cut_mesh_with_vertex_mask(F, to_cut):
    face_to_cut = np.any(to_cut[F], axis=1)
    F = F[~face_to_cut]
    vertex_new_label = np.zeros(to_cut.shape[0], dtype=int)
    vertex_new_label[~to_cut] = np.arange(to_cut.shape[0] - to_cut.sum())
    F = vertex_new_label[F]

    return F, face_to_cut


def construct_geodesic_matrix(V, F, fid):
    res = np.empty((fid.shape[0], fid.shape[0]), dtype=np.float32)
    empty_array = np.array([], dtype=np.int64)
    from tqdm import tqdm
    for i in tqdm(range(fid.shape[0])):
        dist = igl.exact_geodesic(V, F, empty_array, empty_array, fid[i:i+1], fid)
        res[i] = dist
    return res


def calc_geodesic_pair(V, F, pairs):
    res = []
    empty_array = np.array([], dtype=np.int64)
    from tqdm import tqdm
    for i in tqdm(range(len(pairs))):
        dist = igl.exact_geodesic(V, F, empty_array, empty_array, np.array([pairs[i][0]]), np.array([pairs[i][1]]))
        res.append(dist)
    return np.array(res)


def construct_appr_geodesic_matrix(V, F, t=1e-1, on_face=False, n_vert=None):
    res = []
    F = F.astype(np.int64)
    from tqdm import tqdm
    n_vert = V.shape[0] if n_vert is None else n_vert
    loop = tqdm(range(n_vert)) if n_vert > 10 else range(n_vert)
    for i in loop:
        dist = igl.heat_geodesic(V, F, t, np.array([i], dtype=np.int64))
        if on_face:
            dist = dist[F].mean(axis=1)
        res.append(dist)
    res = np.array(res)
    if on_face:
        res = res[F].mean(axis=1)
    return res


def write_objs(vert_pos, faces, obj_path):
    os.makedirs(obj_path, exist_ok=True)
    for i, v in enumerate(vert_pos):
        igl.write_triangle_mesh(os.path.join(obj_path, f'{i:03d}.obj'), v, faces)


def coordinate_transform_cloth3d(vert, invert=False):
    mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    if invert:
        mat = mat.T
    return vert @ mat


def heuristic_boundary_cloth3d(verts, faces, up_axis=2):
    is_boundary = np.zeros(verts.shape[0], dtype=bool)
    boundary = igl.boundary_facets(faces)
    boundary = np.unique(boundary.reshape(-1))
    is_boundary[boundary] = True
    threshold = verts[boundary][:, up_axis].mean()
    return (verts[:, up_axis] > threshold) * is_boundary


@numpy_wrapper
def apply_face_transformation(V, F, J):
    v_all = V[F]
    v_all = v_all.permute(0, 2, 1)
    res = J @ v_all
    res = res.permute(0, 2, 1)
    v_all = v_all.permute(0, 2, 1)
    res *= 0.9
    res = res - res.mean(axis=1, keepdim=True) + v_all.mean(axis=1, keepdim=True)
    res = res.reshape(-1, 3)

    F2 = np.arange(F.shape[0] * 3).reshape(-1, 3)
    return res, F2


@numpy_wrapper
def chamfer_distance(V1, V2):
    V1 = V1.to('cuda:0')
    V2 = V2.to('cuda:0')

    V1 = V1.unsqueeze(0)
    V2 = V2.unsqueeze(1)
    dist = torch.sum((V1 - V2) ** 2, dim=2)
    dist = torch.min(dist, dim=1)[0]
    dist = torch.mean(dist)
    return dist
