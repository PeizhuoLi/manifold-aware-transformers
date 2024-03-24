from torch.autograd import Function
from mesh_utils import csr2torch
import numpy as np
from cholespy import CholeskySolverF, CholeskySolverD, MatrixType
import torch
import igl
from utils import to_tensor


class DifferentiableSolve(Function):
    # Largely inspired by https://github.com/rgl-epfl/large-steps-pytorch/blob/8310713d6a4ab8a88c279f228fd39579a2ca2bcd/largesteps/solvers.py#L128-L148
    """
    Differentiable function to solve the linear system.
    This simply calls the solve methods implemented by the Solver classes.
    """
    @staticmethod
    def forward(ctx, solver, b):
        ctx.solver = solver
        return solver.solve(b)

    @staticmethod
    def backward(ctx, grad_output):
        solver_grad = None  # We have to return a gradient per input argument in forward
        b_grad = None
        if ctx.needs_input_grad[1]:
            b_grad = ctx.solver.solve(grad_output)
        return solver_grad, b_grad


differentiable_solve = DifferentiableSolve.apply


class CholeskySolverDWrapper:
    def __init__(self, *args, **kwargs):
        self.solver = CholeskySolverD(*args, **kwargs)

    def solve(self, b):
        x = torch.zeros_like(b)
        self.solver.solve(b, x)
        return x


class CholeskySolverWrapper:
    def __init__(self, *args, **kwargs):
        self.solver = CholeskySolverF(*args, **kwargs)

    def solve(self, b):
        x = torch.zeros_like(b)
        self.solver.solve(b, x)
        return x


def create_solver(L, device):
    L = L.tocoo()
    n_rows = L.shape[0]
    rows = to_tensor(L.row, device)
    cols = to_tensor(L.col, device)
    data = to_tensor(L.data, device)
    return CholeskySolverWrapper(n_rows, rows, cols, data, MatrixType.COO)


class DiffPoissonSolver:
    def __init__(self, V, F, v_c=None):
        if v_c is None or len(v_c) == 0:
            v_c = [V.shape[0] - 1]
        assert isinstance(v_c, list)
        if isinstance(F, torch.Tensor):
            F = F.cpu().numpy()
        v_c.sort()

        set_v = set(list(range(V.shape[0])))
        set_v_c = set(v_c)
        set_v_f = set_v - set_v_c
        v_f = list(set_v_f)
        v_f.sort()

        self.v_f = v_f
        self.v_c = v_c
        self.F = F
        self.n_vert = V.shape[0]
        self.constraints = V[self.v_c]

        device = V.device

        V_numpy = V.cpu().numpy()

        Lf, Lc = self.create_laplacian(V_numpy)

        G = igl.grad(V_numpy, F)
        self.GT = csr2torch(G.T).to(device)

        area = igl.doublearea(V_numpy, F) / 2
        D_val = np.tile(area[None], (3, 1)).reshape(-1)[:, None]
        self.D_val = to_tensor(D_val, device)

        self.solver = create_solver(Lf, device)
        self.Lc = csr2torch(Lc).to(device)

    def create_laplacian(self, V):
        L = -igl.cotmatrix(V, self.F)
        Lf = L[self.v_f, :][:, self.v_f]
        Lc = L[self.v_f, :][:, self.v_c]
        return Lf, Lc

    def forward(self, M):
        """
        :param M: per face jacobian. F x 3 x 3
        :return:
        """

        if M.ndim > 3:
            M_shape = M.shape
            M = M.reshape((-1,) + M.shape[-3:])
            res = []
            for j in range(M.shape[0]):
                res.append(self.forward(M[j]))
            res = torch.stack(res, dim=0)
            res = res.reshape(M_shape[:-3] + res.shape[-2:])
            return res

        M = M.permute(2, 0, 1)
        M = M.reshape(-1, 3)

        RHS = self.GT @ (self.D_val * M)

        RHS = RHS[self.v_f]
        RHS = RHS - self.Lc @ self.constraints
        res = torch.empty((self.n_vert, 3), device=M.device)
        res[self.v_f] = differentiable_solve(self.solver, RHS)
        res[self.v_c] = self.constraints

        return res
