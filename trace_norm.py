import torch
import numpy as np

def nuclear_norm_grad(x, dy):
    _, U, V = torch.linalg.svd(x, full_matrices=False)
    grad = torch.matmul(U, V.t())
    return dy * grad


def nuclear_norm(x):
    sigma = torch.linalg.svdvals(x)
    norm = torch.sum(sigma)
    return norm


def TensorUnfold(A, k):
    tmp_arr = np.arange(A.dim())
    A = A.permute([tmp_arr[k]] + np.delete(tmp_arr, k).tolist())
    shapeA = A.shape
    A = A.reshape([shapeA[0], np.prod(shapeA[1:])])
    return A


def TensorTraceNorm(X, method='Tucker'):
    shapeX = X.shape
    dimX = len(shapeX)

    if method == 'Tucker':
        re = [nuclear_norm(i) for i in [TensorUnfold(X, j) for j in range(dimX)]]
    elif method == 'TT':
        re = [nuclear_norm(i) for i in
                [X.reshape([np.prod(shapeX[:j]), np.prod(shapeX[j:])]) for j in range(1, dimX)]]
    elif method == 'LAF':
        re = [nuclear_norm(TensorUnfold(X, -1))]
    return torch.stack(re)

