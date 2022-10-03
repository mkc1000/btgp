"""Utilities."""

from math import sqrt
import torch
from torch import Tensor

class TimeOutError(Exception):
    pass

def matvec_deprecated(data_reduced, row_vecs, col_vecs, vector, extra_indices=None):
    """
    TODO: enforce that this is one column (potentially) and do loop outside of the function

    sum_i=1^{n'} u_i v_i^T x_i.

    n' is the number of unique elements (i.e. indices in data reduced).

    where u = row_vecs, v = col_vecs, x=vector.



    data_reduced is n x (d*precision) contains n' unique values


    row_vecs is n x (d*precision): this is a compact representation of the
    of n x n' tensor where there are up to n non-zero elements
    the rows correspond to the data reduced
    col_vecs is n x (d*precision): this is a compact representation of the
    of n x n' tensor where there are up to n non-zero elements
    x is a n x 1 dim tensor

    extra_indices may extend this

    output:
    `n`-dim tensor containing the result which is the same when the entry in data reduced is same
    """
    if extra_indices is None:
        dot_products = torch.zeros_like(vector)
    else:
        dot_products = torch.zeros(
            extra_indices, device=vector.device, dtype=vector.dtype
        )
    out = torch.zeros_like(vector)
    for col in range(data_reduced.shape[1]):
        dot_products.fill_(0)
        dot_products.index_add_(0, data_reduced[:, col], vector * col_vecs[:, col])
        out += dot_products[data_reduced[:, col]] * row_vecs[:, col]

    return out

def matvec(data_reduced, row_vecs, col_vecs, vector, extra_indices=None):
    n, d = data_reduced.shape
    if extra_indices:
        n = extra_indices
    dot_products = torch.zeros(n*d, device=vector.device, dtype=vector.dtype)
    data_reduced = data_reduced + n * torch.arange(d, device=data_reduced.device).view(1, -1)
    vec_col_vecs = col_vecs * vector.view(-1, 1)
    dot_products.index_add_(0, data_reduced.view(-1), vec_col_vecs.view(-1))
    out_mat = dot_products[data_reduced.view(-1)].view(*row_vecs.shape) * row_vecs
    return torch.sum(out_mat, dim=1)

def matvec_vectorizable(data_reduced, row_vecs, col_vecs, vector, extra_indices=None):
    """
    no number can appear in multiple columns of data_reduced
    """
    n, d = data_reduced.shape
    if extra_indices:
        n = extra_indices
    dot_products = torch.zeros(n*d, device=vector.device, dtype=vector.dtype)
    vec_col_vecs = col_vecs * vector.view(-1, 1)
    dot_products.index_add_(0, data_reduced.view(-1), vec_col_vecs.view(-1))
    out_mat = dot_products[data_reduced.view(-1)].view(*row_vecs.shape) * row_vecs
    return torch.sum(out_mat, dim=1)


def matmat(data_reduced, row_vecs, col_vecs, mat):
    dot_products = torch.zeros_like(mat)
    out = torch.zeros_like(mat)
    for col in range(data_reduced.shape[1]):
        dot_products.fill_(0)
        dot_products.index_add_(
            0, data_reduced[:, col], mat * col_vecs[:, col].view(-1, 1)
        )
        out += dot_products[data_reduced[:, col]] * row_vecs[:, col].view(-1, 1)
    return out


def binary(x, bits):
    # Adapted from
    # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=x.device).to(dtype=x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

BIT_TO_BYTE = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)
def bit_array_to_byte_array(arr):
    n_rows = arr.shape[0]
    n_normal_cols = torch.div(arr.shape[1], 8, rounding_mode="floor")
    n_cols = -torch.div(-arr.shape[1], 8, rounding_mode="floor")
    arr_byte_form = torch.empty(n_rows, n_cols, dtype=torch.uint8, device=arr.device)
    arr_byte_form[:, :n_normal_cols] = torch.sum(arr[:, :8*n_normal_cols].view(n_rows, n_normal_cols, 8) * BIT_TO_BYTE.view(1, 1, 8).to(arr), dim=2)
    if n_cols > n_normal_cols:
        extra = torch.zeros(n_rows, 8, dtype=torch.uint8, device=arr.device)
        extra[:, :arr.shape[1]-8*n_normal_cols] = arr[:, 8*n_normal_cols:]
        arr_byte_form[:, -1] = torch.sum(extra * BIT_TO_BYTE.to(arr).view(1, 8), dim=1)
    return arr_byte_form

def matvec_with_just_weights(data_reduced, weights, vector, extra_indices=None):
    """
    This is the same where col_vecs are 1s and row_vecs contains weights times one

    This reduces one element multiplication.
    """

    if extra_indices is None:
        dot_products = torch.zeros_like(vector)
    else:
        dot_products = torch.zeros(
            extra_indices, device=vector.device, dtype=vector.dtype
        )
    out = torch.zeros_like(vector)
    for col in range(data_reduced.shape[1]):
        w = weights[col]
        dot_products.fill_(0)
        dot_products.index_add_(0, data_reduced[:, col], vector * w)
        out += dot_products[data_reduced[:, col]]
    return out

def matvec_with_just_weights_vectorizable(data_reduced, weights, vector, extra_indices=None):
    n, d = data_reduced.shape
    if extra_indices:
        n = extra_indices
    dot_products = torch.zeros(n*d, device=vector.device, dtype=vector.dtype)
    vec_col_vecs = weights.view(1, -1) * vector.view(-1, 1)
    dot_products.index_add_(0, data_reduced.view(-1), vec_col_vecs.view(-1))
    out_mat = dot_products[data_reduced.view(-1)].view(*data_reduced.shape)
    return torch.sum(out_mat, dim=1)

def matmat_with_just_weights(data_reduced, weights, mat, extra_indices=None):
    if extra_indices is None:
        dot_products = torch.zeros_like(mat)
    else:
        dot_products = torch.zeros(
            (extra_indices, mat.shape[1]), device=mat.device, dtype=mat.dtype
        )
    out = torch.zeros_like(mat)
    for col in range(data_reduced.size()[1]):
        dot_products.fill_(0)
        dot_products.index_add_(0, data_reduced[:, col], mat * weights[col])
        out += dot_products[data_reduced[:, col]]
    return out


def rmse(targets: Tensor, predictions: Tensor) -> float:
    """Compute root mean squared error (RMSE).

    Args:
        targets: A `(batch_shape) x n x (m)`-dim tensor of targets.
        predictions: A `(batch_shape) x n x (m)`-dim tensor of predictions.

    Returns:
        A `(batch_shape)`-dim tensor containing the RMSE.
    """
    if targets.ndim < 2 or predictions.ndim < 2:
        return torch.norm(targets - predictions) / sqrt(targets.shape[0])
    return torch.norm(targets - predictions) / sqrt(targets.shape[-2])


def conjgrad(lin_transform, b, x=None, eps=0.001, checkpoint_path=None):
    if x is None:
        x = torch.zeros_like(b)
    else:
        x = x.clone()
    r = b - lin_transform(x)
    p = r.clone()
    rsold = r @ r
    for i in range(len(b)):
        Ap = lin_transform(p)
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        if checkpoint_path is not None:
            torch.save(x, checkpoint_path)
        r = r - alpha * Ap
        rsnew = r @ r
        error = torch.sqrt(rsnew)
        print(
            "alpha", alpha, "error", error
        )  # torch.norm(x - gp.woodbury)) # testing for b = gp.sorted_targets
        if error < eps:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def precon_conjgrad(lin_transform, precon_transform, b, x=None, eps=0.001):
    if x is None:
        x = torch.zeros_like(b)
    else:
        x = x.clone()
    r = b - lin_transform(x)
    z = precon_transform(r)
    p = z.clone()
    rz = r @ z
    for i in range(len(b)):
        Ap = lin_transform(p)
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        error = torch.sqrt(r @ r)
        print(
            "alpha", alpha, "error", error
        )  # torch.norm(x - gp.woodbury)) # testing for b = gp.sorted_targets
        if error < eps:
            break
        z = precon_transform(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x
