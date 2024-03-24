import torch
import numpy as np
import os.path as osp
import random
import torch.nn.functional as F
import igl
from scipy.ndimage import gaussian_filter1d


def batch_mm(matrix, matrix_batch):
    """
    https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)


def numpy_wrapper(func):
    """
    Wrapper for numpy functions that take numpy arrays as input
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        flag = False
        torch_exists = False
        for arg in args:
            if isinstance(arg, np.ndarray):
                flag = True
            if isinstance(arg, torch.Tensor):
                torch_exists = True
        args = [torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
        res = func(*args, **kwargs)
        if flag and not torch_exists:
            if isinstance(res, tuple):
                res = (x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in res)
            else:
                res = res.cpu().numpy() if isinstance(res, torch.Tensor) else res
        return res
    return wrapper


def numpy_wrapper_cuda(func):
    """
    Wrapper for numpy functions that take numpy arrays as input
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        flag = False
        torch_exists = False
        for arg in args:
            if isinstance(arg, np.ndarray):
                flag = True
            if isinstance(arg, torch.Tensor):
                torch_exists = True
        args = [torch.from_numpy(arg).cuda() if isinstance(arg, np.ndarray) else arg for arg in args]
        res = func(*args, **kwargs)
        if flag and not torch_exists:
            if isinstance(res, tuple):
                res = (x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in res)
            else:
                res = res.cpu().numpy() if isinstance(res, torch.Tensor) else res
        return res
    return wrapper


@numpy_wrapper_cuda
def signed_distance(points, vertices, faces, return_libigl=False, extended_vertices=True, winding_number=False):
    
    # in case we want the libigl results, they will be returned here 
    if return_libigl:
        true_S, true_I, true_C = igl.signed_distance(points.cpu().numpy(), vertices.cpu().numpy(), faces.cpu().numpy())
        return true_S, true_I, true_C

    # we can expand the number of vertices by adding the center of each edge to the list of vertices
    # this leads to a better approximation of the closest point on the surface of the mesh
    if extended_vertices:
        vertices =  torch.cat((vertices, 
                    (vertices[faces[:,0]] + vertices[faces[:,2]]) / 2, 
                    (vertices[faces[:,1]] + vertices[faces[:,2]]) / 2), dim=0)

    # in order to identify the closest face to each point, 
    # we compute the distance from each point to the center of each face
    # this works quite well in practices and returns us the face index min_fidx
    average_per_face =  torch.sum(vertices[faces], dim=1)/3
    face_dist = torch.cdist(points, average_per_face)
    _, min_fidx = torch.min(face_dist, dim=1)
    del face_dist

    # identify the sign of the distance to the mesh
    # negative means inside, positive means outside
    # first approach: 
    # winding_number, as detailed in this paper: 
    # https://igl.ethz.ch/projects/winding-number/robust-inside-outside-segmentation-using-generalized-winding-numbers-siggraph-2013-jacobson-et-al.pdf
    # this approach is very slow, but it is the most accurate
    # second approach: 
    # comput the normal of each face, use the matching face indices for each point to identify the normal of the closest face
    # and comput the dot product between the normal and the vector from the point to the center of the face
    # much faster approach, but slightly less accurate
    if winding_number: 
        triangles = vertices[faces].unsqueeze(0).repeat(points.shape[0], 1, 1, 1)
        abc = triangles - points.unsqueeze(1).unsqueeze(1)
        del triangles
        norm = torch.norm(abc, dim=3)
        solid_angle = torch.atan2(  abc[:, :, 0, 0] * abc[:, :, 1, 1] * abc[:, :, 2, 2] +
                                    abc[:, :, 0, 1] * abc[:, :, 1, 2] * abc[:, :, 2, 0] +
                                    abc[:, :, 0, 2] * abc[:, :, 1, 0] * abc[:, :, 2, 1] -
                                    abc[:, :, 0, 2] * abc[:, :, 1, 1] * abc[:, :, 2, 0] -
                                    abc[:, :, 0, 1] * abc[:, :, 1, 0] * abc[:, :, 2, 2] -
                                    abc[:, :, 0, 0] * abc[:, :, 1, 2] * abc[:, :, 2, 1],
                                (torch.sum(abc[:,:,0,:] * abc[:,:,1,:], dim = 2) * norm[:,:,2] +
                                torch.sum(abc[:,:,0,:] * abc[:,:,2,:], dim = 2) * norm[:,:,1] +
                                torch.sum(abc[:,:,1,:] * abc[:,:,2,:], dim = 2) * norm[:,:,0]))
        del abc, norm
        winding = torch.sum(solid_angle, dim=1) / (2 * torch.pi)
        sign = -torch.sign(winding-0.5); del winding, solid_angle
    else: 
        # compute the sign of the distance
        normals = per_face_normals(vertices, faces)
        need_normals = normals[min_fidx]
        product = torch.sum(need_normals * (points - average_per_face[min_fidx]), dim=1)
        sign = torch.sign(product)

    # compute the distance of each point to each vertex (or extended set of vertices)
    dist = torch.cdist(points, vertices)
    min_dist, min_index = torch.min(dist, dim=1)
    del dist

    # default: assign each point to the closest vertex
    p00 = vertices[min_index]

    # implementation of these instructions: 
    # https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    # for each face, we compute whether the closest point on the mesh lies within the triangle 
    # or on one of the edges of the triangle
    # we then identify the closest point there 
    B = vertices[faces[min_fidx, 0]]
    E_0 = vertices[faces[min_fidx, 1]] - vertices[faces[min_fidx, 0]]
    E_1 = vertices[faces[min_fidx, 2]] - vertices[faces[min_fidx, 0]]
    a = torch.sum(E_0 * E_0, dim=1); b = torch.sum(E_0 * E_1, dim=1);    c = torch.sum(E_1 * E_1, dim=1)
    d = torch.sum(E_0 * (vertices[faces[min_fidx, 0]] - points), dim=1); e = torch.sum(E_1 * (vertices[faces[min_fidx, 0]] - points), dim=1)
    det = a * c - b * b; s = (b * e - c * d); t = (b * d - a * e)

    reg0 = (s > 0) * (t > 0) * (s + t < det) # inside the triangle
    reg1 = (s > 0) * (t > 0) * (s + t > det) # outside of the top edge 
    reg3 = (s < 0) * (t > 0) * (s + t < det) # outside of the left edge 
    reg5 = (s > 0) * (t < 0) * (s + t < det) # outside of the bottom edge 
    
    # region0
    s[reg0] = s[reg0] / det[reg0]; t[reg0] = t[reg0] / det[reg0]
    
    # region1
    numer = (c + e) - (b + d);          denom = a - 2 * b + c
    check_numer = numer <= 0;           o_check_numer = numer > 0
    check_denom_numer = numer >= denom; o_check_denom_numer = numer < denom
    s[reg1*check_numer] = 0;            s[reg1*check_denom_numer*o_check_numer] = 1
    s[reg1*o_check_denom_numer*o_check_numer] = (numer[reg1*o_check_denom_numer*o_check_numer] / 
                                                denom[reg1*o_check_denom_numer*o_check_numer])
    t[reg1] = 1 - s[reg1]
    
    # region3
    check_e = e >= 0;       o_check_e = e < 0
    check_e_c = -e >= c;    o_check_e_c = -e < c
    s[reg3] = 0;            t[reg3*check_e] = 0
    t[reg3*o_check_e*check_e_c] = 1
    t[reg3*o_check_e*o_check_e_c] = (-e[reg3*o_check_e*o_check_e_c] / 
                                    c[reg3*o_check_e*o_check_e_c])
    
    #region5
    check_d = d >= 0;       o_check_d = d < 0
    check_d_a = -d >= a;    o_check_d_a = -d < a
    t[reg5] = 0;            s[reg5*check_d] = 0
    s[reg5*o_check_d*check_d_a] = 1
    s[reg5*o_check_d*o_check_d_a] = (-d[reg5*o_check_d*o_check_d_a] / 
                                    a[reg5*o_check_d*o_check_d_a])
    
    # correct closest point for those lying on the edges and inside the triangle
    p00[reg0+reg1+reg3+reg5] = (B[reg0+reg1+reg3+reg5] + s[reg0+reg1+reg3+reg5, None] * E_0[reg0+reg1+reg3+reg5] + 
                                t[reg0+reg1+reg3+reg5, None] * E_1[reg0+reg1+reg3+reg5])
    min_dist[reg0+reg1+reg3+reg5] = torch.norm(p00[reg0+reg1+reg3+reg5] - points[reg0+reg1+reg3+reg5], dim=1)

    return sign*min_dist, min_fidx, p00


# Parameters	V #V by 3 eigen Matrix of mesh vertex 3D positions
#               F #F by 3 eigen Matrix of face (triangle) indices

# Returns	    N #F by 3 eigen Matrix of mesh face (triangle) 3D normals
@numpy_wrapper_cuda
def per_face_normals(vertices, faces):
    face_normals = torch.cross(vertices[faces[:,1]] - vertices[faces[:,0]], vertices[faces[:,2]] - vertices[faces[:,0]])
    face_normals = face_normals / torch.norm(face_normals, dim=1).unsqueeze(1)
    return face_normals


def varmean2stdmean(var_mean):
    if isinstance(var_mean[0], tuple):
        var_mean = [np.concatenate([x[i] for x in var_mean]) for i in range(2)]
    res = [var_mean[0] ** 0.5 + 1e-10, var_mean[1]]
    if isinstance(res[0], np.ndarray):
        res = [torch.from_numpy(r) for r in res]
    return res


def var_mean_list(lst, last_dim=None):
    if isinstance(lst, np.ndarray):
        lst = lst.reshape(-1, last_dim)
        return np.var(lst, axis=0), np.mean(lst, axis=0)
    if last_dim is None:
        last_dim = lst[0].shape[-1]
    ex = np.array([np.mean(l.astype(np.float64).reshape(-1, last_dim), axis=0) for l in lst])
    cnt = np.array([l.size // last_dim for l in lst])[:, None]
    weight = cnt.astype(np.float64) / cnt.sum()

    e_total = (weight * ex).sum(axis=0)

    varx = np.array([np.mean((l.astype(np.float64).reshape(-1, last_dim) - e_total) ** 2, axis=0) for l in lst])
    var_total = (weight * varx).sum(axis=0)

    return var_total.astype(lst[0].dtype), e_total.astype(lst[0].dtype)


def get_std_mean_fast(dataset):
    print('Calculating var and mean with fast approximation')
    if not hasattr(dataset, 'datasets'):
        jacobians = dataset.jacobians[dataset.cfg.cond_length - 1:]
        relative_stretch = dataset.stretch_base
        orients = dataset.orients_base
        vert_pos = dataset.centroids_base
        global_velo = dataset.global_velo
        sdf = dataset.sdf.transpose(1, 2, 3, 0)
        base_deformations = dataset.base_deformations if dataset.datasets[0].base_deformations is not None else None
        face_velo_input = dataset.face_velo_input
        singular_value = dataset.singular_value
        body_g_velo = dataset.body_g_velo
    else:
        jacobians = [d.jacobians for d in dataset.datasets]
        relative_stretch = [d.stretch_base for d in dataset.datasets]
        orients = [d.orients_base for d in dataset.datasets] if dataset.datasets[0].orients_base is not None else None
        vert_pos = [d.centroids_base for d in dataset.datasets]
        global_velo = [d.global_velo for d in dataset.datasets] if dataset.datasets[0].global_velo is not None else None
        sdf = [d.sdf.transpose(1, 2, 3, 0) for d in dataset.datasets] if dataset.datasets[0].sdf is not None else None
        base_deformations = [d.base_deformations for d in dataset.datasets] if dataset.datasets[0].base_deformations is not None else None
        singular_value = [d.singular_value for d in dataset.datasets] if dataset.datasets[0].singular_value is not None else None
        body_g_velo = [d.body_g_velo for d in dataset.datasets] if dataset.datasets[0].body_g_velo is not None else None

    # prepare var mean for input
    var_mean_inputs = []
    std_mean_dict_input = {}

    if dataset.cfg.use_relative_stretch:
        var_mean_relative_stretch = var_mean_list(relative_stretch, last_dim=9)
        var_mean_inputs.append(var_mean_relative_stretch)

    if orients is not None:
        var_mean_orients = var_mean_list(orients, last_dim=orients[0].shape[-1])
        var_mean_inputs.append(var_mean_orients)

    var_mean_vert_pos = var_mean_list(vert_pos, last_dim=3)
    var_mean_vert_pos[0][:] *= 2
    var_mean_vert_pos[1][:] = 0
    var_mean_inputs.append(var_mean_vert_pos)

    # prepare var mean for sdf
    if dataset.cfg.use_sdf:
        if dataset.cfg.normalize_sdf:
            last_dim_sdf = sdf[0].shape[-2] if isinstance(sdf, list) else sdf.shape[-2]

            if last_dim_sdf == 8:
                var_mean_sdf_all = var_mean_list(sdf, last_dim=8)
                if dataset.cfg.use_sdf == 1:
                    var_mean_sdf = [var_mean_sdf_all[0][::2], var_mean_sdf_all[1][::2]]
                else:
                    var_mean_sdf = var_mean_sdf_all
            else:
                var_mean_sdf = var_mean_list(sdf, last_dim=4)
        else:
            n_dim = 4 * dataset.cfg.use_sdf
            var_mean_sdf = (np.ones((n_dim,), dtype=np.float32), np.zeros((n_dim,), dtype=np.float32))
        var_mean_inputs.append(var_mean_sdf)

    if dataset.cfg.add_base_deformation:
        if dataset.cfg.normalize_base_deformation:
            var_mean_base = var_mean_list(base_deformations, last_dim=12)
            var_mean_inputs.append(var_mean_base)
        else:
            var_mean_base = (np.ones((12,), dtype=np.float32), np.zeros((12,), dtype=np.float32))
            var_mean_inputs.append(var_mean_base)

    if dataset.cfg.predict_singular_value:
        var_mean_singular_value = var_mean_list(singular_value, last_dim=3)
        var_mean_inputs.append(var_mean_singular_value)

    # prepare var mean for channels that do not require normalization
    delta_dim = dataset.cfg.n_channel - sum([x[0].shape[-1] for x in var_mean_inputs])
    var_mean_extra = (np.ones((delta_dim,), dtype=np.float32), np.zeros((delta_dim,), dtype=np.float32))
    var_mean_inputs.append(var_mean_extra)

    var_mean_input = [np.concatenate([x[i] for x in var_mean_inputs]) for i in range(2)]

    std_mean_dict_input['f'] = varmean2stdmean(var_mean_input)

    # prepare var mean for output
    var_mean_outputs = []

    var_mean_jacobians = var_mean_list(jacobians, last_dim=9)
    var_mean_outputs.append(var_mean_jacobians)

    if dataset.cfg.predict_singular_value:
        var_mean_outputs.append(var_mean_singular_value)

    var_mean_output = [np.concatenate([x[i] for x in var_mean_outputs]) for i in range(2)]

    std_mean_dict_output = {'f': varmean2stdmean(var_mean_output)}

    if global_velo is not None and dataset.cfg.predict_g_velo:
        var_mean_global_velo = var_mean_list(global_velo, last_dim=3)
        std_mean_dict_output['g_velo'] = varmean2stdmean(var_mean_global_velo)

    if global_velo is not None and dataset.cfg.predict_g_velo:
        if body_g_velo is not None:
            var_mean_body_g_velo = var_mean_list(body_g_velo, last_dim=3)
            var_mean_global_velo = [var_mean_global_velo, var_mean_body_g_velo]
            var_mean_global_velo = [np.concatenate([x[i] for x in var_mean_global_velo]) for i in range(2)]

        std_mean_dict_input['g_velo'] = varmean2stdmean(var_mean_global_velo)

    return std_mean_dict_input, std_mean_dict_output


def get_std_mean_fast2(dataset, multiple_dataset=False):
    print('Calculating var and mean with fast approximation')
    if not multiple_dataset:
        jacobians = dataset.jacobians[dataset.cond_length - 1:]
        relative_stretch = dataset.stretch_base
        vert_pos = dataset.centroids_base
        face_velo = dataset.face_velo
        global_velo = dataset.global_velo
    else:
        jacobians = np.concatenate([d.jacobians for d in dataset.datasets], axis=0)
        relative_stretch = np.concatenate([d.stretch_base for d in dataset.datasets], axis=0)
        vert_pos = np.concatenate([d.centroids_base for d in dataset.datasets], axis=0)
        face_velo = np.concatenate([d.face_velo for d in dataset.datasets], axis=0) if dataset.datasets[0].face_velo is not None else None
        global_velo = np.concatenate([d.global_velo for d in dataset.datasets], axis=0) if dataset.datasets[0].global_velo is not None else None

    # prepare var mean for output
    var_mean_outputs = []

    var_mean_jacobians = np.var(jacobians.reshape(-1, 9), axis=0), np.mean(jacobians.reshape(-1, 9), axis=0)
    var_mean_outputs.append(var_mean_jacobians)

    if face_velo is not None:
        var_mean_face_velo = np.var(face_velo.reshape(-1, 3), axis=0), np.mean(face_velo.reshape(-1, 3), axis=0)
        var_mean_outputs.append(var_mean_face_velo)

    var_mean_output = [np.concatenate([x[i] for x in var_mean_outputs]) for i in range(2)]

    std_mean_dict_output = {'f': varmean2stdmean(var_mean_output)}

    if global_velo is not None:
        var_mean_global_velo = np.var(global_velo.reshape(-1, 3), axis=0), np.mean(global_velo.reshape(-1, 3), axis=0)
        std_mean_dict_output['g_velo'] = varmean2stdmean(var_mean_global_velo)

    # prepare var mean for input
    var_mean_inputs = []

    if dataset.use_relative_stretch:
        var_mean_relative_stretch = np.var(relative_stretch.reshape(-1, 9), axis=0), np.mean(relative_stretch.reshape(-1, 9), axis=0)
        var_mean_inputs.append(var_mean_relative_stretch)

    var_mean_vert_pos = (2 * np.var(vert_pos.reshape(-1, 3), axis=0), np.zeros((3,), dtype=np.float32))
    var_mean_inputs.append(var_mean_vert_pos)

    # prepare var mean for channels that do not require normalization
    delta_dim = dataset.n_channel - sum([x[0].shape[-1] for x in var_mean_inputs])
    var_mean_extra = (np.ones((delta_dim,), dtype=np.float32), np.zeros((delta_dim,), dtype=np.float32))
    var_mean_inputs.append(var_mean_extra)

    var_mean_input = [np.concatenate([x[i] for x in var_mean_inputs]) for i in range(2)]

    std_mean_dict_input = {'f': varmean2stdmean(var_mean_input)}

    if global_velo is not None:
        std_mean_dict_input['g_velo'] = std_mean_dict_output['g_velo']

    return std_mean_dict_input, std_mean_dict_output


def get_std_mean(dataset, use_tqdm=False, threshold=int(5e5)):
    all_input = []
    all_output = []

    if len(dataset) < threshold:
        loop_items = range(len(dataset))
    else:
        loop_items = np.random.choice(len(dataset), threshold, replace=False)

    print('Calculating var and mean')
    if use_tqdm:
        from tqdm import tqdm
        loop = tqdm(loop_items)
    else:
        loop = loop_items

    step_size = len(loop) / 20
    for cnt, i in enumerate(loop):
        if cnt % step_size == 0:
            print('Progress: {:.2f}%'.format(cnt / len(loop) * 100))
        input, output = dataset[i]
        all_input.append(input)
        all_output.append(output)

    all_input = torch.cat(all_input, dim=0)
    all_output = torch.stack(all_output, dim=0)

    all_input = all_input.reshape(-1, all_input.shape[-1])
    all_output = all_output.reshape(-1, all_output.shape[-1])

    var_mean_input = torch.var_mean(all_input, dim=0)
    var_mean_output = torch.var_mean(all_output, dim=0)

    if dataset.use_handle_indicator:
        var_mean_input[0][-1] = 1.
        var_mean_input[1][-1] = 0.

    return (var_mean_input[0] ** 0.5, var_mean_input[1]), (var_mean_output[0] ** 0.5, var_mean_output[1])


def warm_up_incremental(dataset, s_input, s_output, threshold=int(5e5), use_tqdm=False):
    loop_items = np.random.choice(len(dataset), threshold, replace=False)

    print('Calculating var and mean')
    if use_tqdm:
        from tqdm import tqdm
        loop = tqdm(loop_items)
    else:
        loop = loop_items

    step_size = len(loop) / 20
    for cnt, i in enumerate(loop):
        if cnt % step_size == 0 and not use_tqdm:
            print('Progress: {:.2f}%'.format(cnt / len(loop) * 100))
        input, output = dataset[i]
        s_input.update(input)
        s_output.update(output)


class IncrementalStatistics:
    def __init__(self, mean=0, var=1, n_samples=0, keeplast=0, eps=1e-9):
        self.ex = torch.tensor(mean + 0., dtype=torch.float64)
        self.ex2 = torch.tensor(var + mean ** 2, dtype=torch.float64)
        self.n_samples = n_samples
        self.keeplast = keeplast
        self.eps = eps

    def update(self, new_val):
        new_val = new_val.reshape(-1, new_val.shape[-1]).to(torch.float64)
        new_samples = new_val.shape[0]
        new_mean = new_val.mean(0)
        new_mean2 = (new_val ** 2).mean(0)

        self.ex = self.n_samples / (self.n_samples + new_samples) * self.ex + new_samples / (self.n_samples + new_samples) * new_mean
        self.ex2 = self.n_samples / (self.n_samples + new_samples) * self.ex2 + new_samples / (self.n_samples + new_samples) * new_mean2
        self.n_samples += new_samples

        if self.keeplast:
            self.ex[-self.keeplast:] = 0.
            self.ex2[-self.keeplast:] = 1.

    def __getitem__(self, item):
        if item == 0:
            res = (torch.clamp_min(self.ex2 - self.ex ** 2, 0.) + self.eps) ** 0.5
        elif item == 1:
            res = self.ex
        else:
            raise IndexError
        return res.to(torch.float32)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_noise_level_from_args(args, dataset):
    noise_level = args.noise_level
    if noise_level > 0.9:
        noise_level = torch.ones((args.cond_length + args.dai_extra, dataset.n_channel), device=args.device) * noise_level
        if args.handle_indicator:
            noise_level[:, -1] = 0.
        noise_level = noise_level.reshape(-1)
    return noise_level


def checkpoint_sort_key(x):
    base = 10000000
    x = x.split('.')[0]
    if x.count('_') == 1:
        return int(x.split('_')[1]) * base
    else:
        return int(x.split('_')[1]) * base + int(x.split('_')[2])


def to_device(d, device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)


def reshape_past(in_dict):
    for k, v in in_dict.items():
        if k in ['f', 'g_velo']:
            in_dict[k] = v.reshape(v.shape[:-2] + (-1,))


def get_unique_name(prefix, name):
    base_name, suffix = name.split('.')
    while True:
        new_name = osp.join(prefix, f'{base_name}_{hex(random.randint(0, 2**64))}.{suffix}')
        if not osp.exists(new_name):
            return new_name


def to_tensor(v, device='cpu'):
    if isinstance(v, np.ndarray):
        if v.dtype == np.float64:
            return torch.tensor(v, dtype=torch.float32, device=device)
        else:
            return torch.from_numpy(v).to(device)
    elif isinstance(v, torch.Tensor):
        return v.to(device)
    elif v is None:
        return None
    else:
        raise ValueError('Unknown type: {}'.format(type(v)))


class DynamicMmap:
    def __init__(self, filename, dtype, mode, shape):
        self.filename = filename
        self.dtype = dtype
        self.mode = mode
        self.shape = shape

    def get_kernel(self):
        return np.memmap(self.filename, dtype=self.dtype, mode=self.mode, shape=self.shape)

    def __getattr__(self, item):
        if item in ['shape', 'dtype', 'mode', 'filename', 'get_kernel']:
            return getattr(self, item)
        else:
            kernel = self.get_kernel()
            return getattr(kernel, item)

    def __getitem__(self, item):
        kernel = self.get_kernel()
        return kernel[item]


@numpy_wrapper
def interpolate_t(x, target_length, mode, reverse=False, axis=None, source_length=None, make_static=False, start_frame=0):
    """
    1D Interpolate along a given axis with given mode
    :param x:
    :param target_length:
    :param axis:
    :return:
    """
    if axis is None:
        if source_length is None:
            raise Exception('Either axis or source_length must be given')
        axis = list(x.shape).index(source_length)
    last_dim = x.ndim - 1
    x = x.transpose(axis, last_dim)
    xshape = list(x.shape)
    x = x.reshape(1, -1, x.shape[-1])
    if make_static:
        x[:] = x[..., :1]
    x = x[..., start_frame:]
    if x.shape[-1] != target_length:
        xshape[-1] = target_length
        ori_type = x.dtype
        x = x.to(torch.float32)
        x = F.interpolate(x, target_length, mode=mode)
        x = x.to(ori_type)
    if reverse:
        x = torch.flip(x, [-1])
    x = x.reshape(xshape)
    x = x.transpose(last_dim, axis)
    return x


@numpy_wrapper
def smooth_t(x, sigma, axis=None, source_length=None):
    """
    1D Interpolate along a given axis with given mode
    :param x:
    :param sigma:
    :param axis:
    :return:
    """
    if axis is None:
        if source_length is None:
            raise Exception('Either axis or source_length must be given')
        axis = list(x.shape).index(source_length)
    last_dim = x.ndim - 1
    x = x.transpose(axis, last_dim)
    xshape = list(x.shape)
    x = x.reshape(1, -1, x.shape[-1])

    x = x.numpy()
    x = gaussian_filter1d(x, sigma=sigma, axis=-1)
    x = torch.from_numpy(x)

    x = x.reshape(xshape)
    x = x.transpose(last_dim, axis)
    return x


@numpy_wrapper
def auto_cut(x, cut_mask):
    n_target = cut_mask.shape[0]
    axis = list(x.shape).index(n_target)
    x = x.transpose(axis, 0)
    x = x[~cut_mask]
    x = x.transpose(0, axis)
    return x


class BatchedMultipleDatasetSampler:
    def __init__(self, data_source, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.total_length = 0
        self.size_map = {}
        self.shuffle = shuffle
        for data in data_source.datasets:
            sample_size = data.sample_size
            if sample_size not in self.size_map:
                self.size_map[sample_size] = []
            self.size_map[sample_size] += list(range(self.total_length, self.total_length + len(data)))
            self.total_length += len(data)

    def __iter__(self):
        batches = []
        for key in self.size_map:
            new_seq = self.size_map[key].copy()
            if self.shuffle:
                random.shuffle(new_seq)
            new_seq = new_seq[:len(new_seq) // self.batch_size * self.batch_size]
            for i in range(0, len(new_seq), self.batch_size):
                batches.append(new_seq[i:i + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self):
        return self.total_length


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@numpy_wrapper
def repr6d2mat(repr):
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat


@numpy_wrapper
def matrix_from_diag(diag):
    res = torch.zeros(diag.shape + (diag.shape[-1],), dtype=diag.dtype, device=diag.device)
    res[..., torch.arange(diag.shape[-1]), torch.arange(diag.shape[-1])] = diag
    return res


@numpy_wrapper
def inv_rot_mat(mat):
    return mat.transpose(-1, -2)


def dict_to_csv(dic):
    all_str = ''
    all_str += 'name, value\n'
    for l in dic:
        all_str += '{}, {}\n'.format(l[0], l[1])
    return all_str


def create_stress_test(in_dict, n):
    keys = ['f', 'geodesic']
    n_ori = in_dict['f'].shape[1]
    for key in keys:
        if key in in_dict:
            shape = in_dict[key].shape
            new_shape = tuple(n if j == n_ori else j for j in shape)
            val2 = torch.randn(new_shape, device=in_dict[key].device, dtype=in_dict[key].dtype)
            in_dict[key] = val2
    return in_dict


def detect_abnormal(parameters):
    for p in parameters:
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            return True
    return False


class AbnormalDetector:
    def __init__(self, parameters):
        self.parameters = parameters

    def detect(self):
        return detect_abnormal(self.parameters)
