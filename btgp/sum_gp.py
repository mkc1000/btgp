import numpy as np
import torch
from efficient_gp.utils import (
    conjgrad,
    matvec,
    matvec_with_just_weights,
    matmat_with_just_weights,
    precon_conjgrad,
)


class KernelSumGP(object):
    def __init__(self, gp_population):
        self.gps = gp_population.gps
        self.n_gps = len(self.gps)
        self.gp = gp_population.gp
        self.device = self.gp.device
        self.nlls = torch.tensor(gp_population.nlls, device=self.device)
        self.weights = torch.ones_like(self.nlls)
        self.weights /= torch.sum(self.weights)
        self.lamb = self.gp.lamb
        if self.gp.data_bit_array is None:
            self.gp.make_data_bit_array()
        self.precon_basis_vecs = None
        self.inv_in_precon_basis = None
        self.small_eigspace_grps = None
        self.small_precon_diag_part = None
        self.small_precon_remove_principal = None
        self.woodbury = None

    def get_small_eigenspaces(self):
        eig_space_grps = None
        eigenvalues = torch.empty(
            (self.gp.train_n, self.n_gps),
            device=self.gp.device,
            dtype=self.gps[0].weights.dtype,
        )
        for i, (gp, weight) in enumerate(zip(self.gps, self.weights)):
            gp.modify_gp(self.gp)
            self.gp.make_data_reduced()
            if eig_space_grps is None:
                eig_space_grps = torch.empty(
                    (self.gp.train_n, self.n_gps),
                    device=self.gp.device,
                    dtype=self.gp.data_reduced.dtype,
                )
            grp_ids, eigvals, _ = self.gp.small_eigenspaces()
            eig_space_grps[:, i] = grp_ids
            eigenvalues[:, i] = weight * eigvals
        _, grp_mates_all_gps, counts = torch.unique(
            eig_space_grps, return_inverse=True, return_counts=True, dim=0
        )
        multiplicities = counts[grp_mates_all_gps] - 1
        return grp_mates_all_gps, multiplicities, torch.sum(eigenvalues, dim=1)

    def small_eigvec_precon(self):
        grp_mates_all_gps, multiplicities, eigenvalues = self.get_small_eigenspaces()
        # first multiply by identity per group
        elemwise_mult = torch.ones_like(eigenvalues) + (
            1 / (eigenvalues + 1e-9) * (multiplicities > 0)
        )
        second_mult = (
            (multiplicities > 0)
            / torch.sqrt(eigenvalues + 1e-9)
            / torch.sqrt(multiplicities + 1)
        )
        # We'll do vec * elemwise_mult and then subtract matvec(grp_mates_all_gps, second_mult, second_mult, vec)
        self.small_eigspace_grps = grp_mates_all_gps
        self.small_precon_diag_part = elemwise_mult
        self.small_precon_remove_principal = second_mult

    def precon_transform_small(self, vector):
        return vector * self.small_precon_diag_part - matvec(
            self.small_eigspace_grps.view(-1, 1),
            self.small_precon_remove_principal.view(-1, 1),
            self.small_precon_remove_principal.view(-1, 1),
            vector,
        )

    def precon_transform_small_matmat(self, mat):
        return mat * self.small_precon_diag_part.view(-1, 1) - matmat(
            self.small_eigspace_grps.view(-1, 1),
            self.small_precon_remove_principal.view(-1, 1),
            self.small_precon_remove_principal.view(-1, 1),
            mat,
        )

    def get_preconditioner(
        self, depth=3
    ):  # THIS DOES WORSE THAN get_preconditioner_brute
        for i, (gp, weight) in enumerate(zip(self.gps, self.weights)):
            gp.modify_gp(self.gp)
            self.gp.make_data_reduced()
            low_rank, diag = self.gp.low_rank(depth=depth)
            if i == 0:
                vecs = low_rank
                D = diag * weight
            else:
                vecs = torch.hstack((vecs, low_rank))
                D = torch.concat((D, diag * weight))
        Q, R = torch.linalg.qr(vecs)
        S, U = torch.linalg.eigh(R @ torch.diag(D) @ R.t())
        Q_ = (Q @ U)[:, S > 2]
        S_ = S[S > 2]
        self.precon_basis_vecs = Q_
        self.inv_in_precon_basis = torch.diag(1 / S_ - 1)
        return self.precon_basis_vecs, self.inv_in_precon_basis

    def get_preconditioner_brute(self, rank=50, n_iter=15, verbose=True):
        vecs = torch.normal(
            0,
            1,
            (self.gp.train_n, rank),
            device=self.device,
            dtype=self.gps[0].weights.dtype,
        )
        vecs[:, 0] = 1.0
        vecs, R = torch.linalg.qr(vecs)
        for _ in range(n_iter):
            vecs = self.kernel_matmat(vecs)
            vecs, R = torch.linalg.qr(vecs)
            diag = torch.abs(torch.diagonal(R))
            order = torch.argsort(diag, descending=True)
            vecs = vecs.t()[order].t()
            if verbose:
                print(diag)
        self.precon_basis_vecs = vecs
        self.inv_in_precon_basis = torch.diag(1 / diag - 1)

    def kernel_matvec(self, vector):
        out = torch.zeros_like(vector)
        for gp, weight in zip(self.gps, self.weights):
            gp.modify_gp(self.gp)
            self.gp.make_data_reduced()
            out += weight * matvec_with_just_weights(
                self.gp.data_reduced[: self.gp.train_n],
                self.gp.weights,
                vector,
                extra_indices=self.gp.total_n,
            )
        out += self.lamb * vector
        return out

    def precon_transform(self, vector):
        return vector + self.precon_basis_vecs @ (
            self.inv_in_precon_basis @ (self.precon_basis_vecs.t() @ vector)
        )

    def kernel_matmat(self, mat):
        out = torch.zeros_like(mat)
        for gp, weight in zip(self.gps, self.weights):
            gp.modify_gp(self.gp)
            self.gp.make_data_reduced()
            out += weight * matmat_with_just_weights(
                self.gp.data_reduced[: self.gp.train_n],
                self.gp.weights,
                mat,
                extra_indices=self.gp.total_n,
            )
        out += self.lamb * mat
        return out

    def precon_matmat(self, mat):
        return self.precon_transform(mat)

    def double_precon_transform(self, vector):
        return self.precon_transform_small(self.precon_transform(vector))

    def calculate_woodbury(self, tol=1.0, init_guess=None):
        if self.precon_basis_vecs is None:
            self.woodbury = conjgrad(
                self.kernel_matvec, self.gp.targets, x=init_guess, eps=tol
            )
        else:
            if self.small_eigspace_grps is None:
                self.woodbury = precon_conjgrad(
                    self.kernel_matvec,
                    self.precon_transform,
                    self.gp.targets,
                    x=init_guess,
                    eps=tol,
                )
            else:
                self.woodbury = precon_conjgrad(
                    self.kernel_matvec,
                    self.double_precon_transform,
                    self.gp.targets,
                    x=init_guess,
                    eps=tol,
                )

    def predict_means(self):
        self.woodbury_extended = torch.cat(
            (
                self.woodbury,
                torch.zeros(
                    self.gp.total_n - self.gp.train_n,
                    device=self.woodbury.device,
                    dtype=self.woodbury.dtype,
                ),
            )
        )
        out = torch.zeros_like(self.gp.woodbury_extended[self.gp.train_n :])
        for gp, weight in zip(self.gps, self.weights):
            gp.modify_gp(self.gp)
            self.gp.make_data_reduced()
            out += (
                weight
                * matvec_with_just_weights(
                    self.gp.data_reduced, self.gp.weights, self.woodbury_extended
                )[self.gp.train_n :]
            )
        return out

    def rmse(self, targets):
        return torch.sqrt(torch.mean(torch.square(targets - self.predict_means())))
