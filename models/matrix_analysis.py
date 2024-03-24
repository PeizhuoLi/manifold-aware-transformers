import numpy as np
import torch


def batched_svd_torch(F):
    F_shape = F.shape
    F = F.reshape(-1, 3, 3)

    U, sigma, Vt = torch.linalg.svd(F)
    flag = torch.det(U) < 0
    U[flag, :, 2] *= -1
    sigma[flag, 2] *= -1

    flag = torch.det(Vt) < 0
    Vt[flag, 2, :] *= -1
    sigma[flag, 2] *= -1

    U = U.reshape(F_shape)
    sigma = sigma.reshape(F_shape[:-1])
    Vt = Vt.reshape(F_shape)

    return U, sigma, Vt


def batched_svd(F):
    F_shape = F.shape
    F = F.reshape(-1, 3, 3)

    U, sigma, Vt = np.linalg.svd(F)
    flag = np.linalg.det(U) < 0
    U[flag, :, 2] *= -1
    sigma[flag, 2] *= -1

    flag = np.linalg.det(Vt) < 0
    Vt[flag, 2, :] *= -1
    sigma[flag, 2] *= -1

    U = U.reshape(F_shape)
    sigma = sigma.reshape(F_shape[:-1])
    Vt = Vt.reshape(F_shape)

    return U, sigma, Vt


def clamp_singular_value(F, minv=0.4, maxv=1.4, val=None, use_gpu=False):
    if minv is None and maxv is None and val is None:
        return F

    if use_gpu:
        F = F.cuda()
        U, sigma, Vt = batched_svd_torch(F)
        if val is not None:
            sigma[:] = val
        else:
            sigma = np.clip(sigma, minv, maxv)

        sigma_mat = torch.zeros(sigma.shape + (3,), dtype=torch.float32, device=F.device)
        sigma_mat[..., 0, 0] = sigma[..., 0]
        sigma_mat[..., 1, 1] = sigma[..., 1]
        sigma_mat[..., 2, 2] = sigma[..., 2]
        return U @ sigma_mat @ Vt

    else:
        U, sigma, Vt = batched_svd(F)
        if val is not None:
            sigma[:] = val
        else:
            sigma = np.clip(sigma, minv, maxv)

        sigma_mat = np.zeros(sigma.shape + (3,), dtype=np.float32)
        sigma_mat[..., 0, 0] = sigma[..., 0]
        sigma_mat[..., 1, 1] = sigma[..., 1]
        sigma_mat[..., 2, 2] = sigma[..., 2]
        return U @ sigma_mat @ Vt
