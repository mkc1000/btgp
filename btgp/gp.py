import math
import torch
import numpy as np
from efficient_gp.utils import binary, matvec, matvec_with_just_weights, matvec_vectorizable, matvec_with_just_weights_vectorizable, rmse, TimeOutError, bit_array_to_byte_array
from efficient_gp.bfgs import BFGS
from scipy.optimize import minimize, Bounds, basinhopping
import os
import errno

# Encoding bit order and weights with single parameter vector:
# (nll is continuous and piecewise smooth w.r.t. this parameter vector)

# To go from bit order and weights to vector:
# vec = boxize(bit_order_to_permutation(bit_order, precision), weights)
# or call gp.bit_order_and_weights_to_param_vec(), and result will be stored in gp.param_vec
# To go from vector to bit_order and weights:
# perm, weights = vec_to_perm_and_weights(vec)
# bit_order = permutation_to_bit_order(perm, precision)
# or call gp.param_vec_to_bit_order_and_weights(), and results will be stored in gp.bit_order and gp.weights

def bit_order_to_permutation(bit_order, precision):
    return bit_order[:, 1] * precision + bit_order[:, 0]

def bit_order_to_inv_permutation(bit_order, precision):
    perm = bit_order_to_permutation(bit_order, precision)
    return torch.argsort(perm)

def permutation_to_bit_order(perm, precision):
    bit = torch.div(perm, precision, rounding_mode="floor")
    depth = perm - bit * precision
    return torch.hstack((depth.view(-1, 1), bit.view(-1, 1)))

def boxize_old(permutation, weights):
    weights_flipped = torch.flip(weights, dims=(-1,))
    weight_to_go_flipped = torch.cumsum(weights_flipped, dim=-1)
    weight_to_go = torch.flip(weight_to_go_flipped, dims=(-1,))
    vec = torch.empty_like(weights)
    vec[permutation] = weight_to_go
    return vec

def boxize(inv_permutation, weights):
    weights_flipped = torch.flip(weights, dims=(-1,))
    weight_to_go_flipped = torch.cumsum(weights_flipped, dim=-1)
    weight_to_go = torch.flip(weight_to_go_flipped, dims=(-1,))
    return weight_to_go[inv_permutation]

def sanitize_vec(vec, precision):
    row_for_each_dim = vec.view(-1, precision)
    row_for_each_dim_sorted, reorderings = torch.sort(row_for_each_dim, dim=1, descending=True)
    return vec, reorderings # something is not working properly, so this function is overridden
    return row_for_each_dim_sorted.view(-1), reorderings

def vec_to_perm_and_weights(vec):
    weight_to_go, perm = torch.sort(vec, descending=True) # when there are ties, tie-breaking behavior is undefined (random?)
    weights = -torch.diff(weight_to_go, append=torch.tensor((0,), dtype=weight_to_go.dtype, device=weight_to_go.device))
    return perm, torch.abs(weights) # abs is only here because some weights are -0.0

def grad_to_vec_grad_old(permutation, weight_grad):
    vec_grad = torch.empty_like(weight_grad)
    grad_diff = torch.diff(weight_grad, prepend=torch.tensor((0.,), dtype=weight_grad.dtype, device=weight_grad.device))
    vec_grad[permutation] = grad_diff
    return vec_grad

def grad_to_vec_grad(inv_permutation, weight_grad):
    grad_diff = torch.diff(weight_grad, prepend=torch.tensor((0.,), dtype=weight_grad.dtype, device=weight_grad.device))
    vec_grad = grad_diff[inv_permutation]
    return vec_grad

def rn_to_box(vec):
    vec -= torch.clamp(torch.min(vec), max=0)
    return vec / torch.max(vec)

def extra_loss(unbounded_vec, c=1):
    min_too_small = torch.square(torch.clamp(torch.min(unbounded_vec), max=0))
    max_not_one = torch.square(torch.max(unbounded_vec) - 1)
    return c * (min_too_small + max_not_one)

def box_vec_grad_to_rn_vec_grad(box_vec_grad, rn_vec, box_vec, c=1):
    min_or_zero = torch.clamp(torch.min(rn_vec), max=0)
    max = torch.max(rn_vec)
    vec_grad = box_vec_grad / (max - min_or_zero)
    mask = rn_vec == max
    n_way_tie = torch.sum(mask)
    vec_grad[mask] += -(box_vec_grad @ box_vec) / (max - min_or_zero) / n_way_tie
    vec_grad[mask] += 2 * (max - 1) * c / n_way_tie
    if min_or_zero <= 0:
        mask = rn_vec == min_or_zero
        n_way_tie = torch.sum(mask)
        vec_grad[mask] += -(box_vec_grad @ (1 - box_vec)) / (max - min_or_zero) / n_way_tie
        vec_grad[mask] += 2 * min_or_zero * c / n_way_tie
    return vec_grad

def vec_grad_sanitized_to_unsanitized(vec_grad, reorderings):
    return vec_grad # something is not working properly, so this function is overridden
    vectorized_reorderings = (reorderings + reorderings.shape[1]*torch.arange(reorderings.shape[0], device=reorderings.device).view(-1, 1)).view(-1)
    vec_grad_unsan = torch.empty_like(vec_grad)
    vec_grad_unsan[vectorized_reorderings] = vec_grad
    return vec_grad_unsan


class ONGP(object):
    def __init__(self, train_data, targets, test_data, precision='auto',
                 lambd=0.0001, default_lr=0.01, min_weight=0.0):
        self.device = train_data.device
        self.torch_type = train_data.dtype
        self.train_n, self.dim = train_data.size()
        if precision == 'auto':
            self.precision = (150 // self.dim) + 1
            if self.precision > 8:
                self.precision = 8
        else:
            self.precision = precision
        self.data = torch.vstack((train_data, test_data)) # memory suggestion: after initializing ONGP, del train_data & test_data
        self.data = torch.clip(self.data, 0, 1-2**(-self.precision))
        self.total_n = self.data.size()[0]
        self.targets = targets
        self.lamb = lambd
        self.min_weight = min_weight
        self.active_bit_slice = slice(0, self.dim * self.precision, 1)
        self.bit_order = torch.stack(torch.meshgrid(torch.arange(self.precision, device=self.device, dtype=torch.int64), torch.arange(self.dim, device=self.device, dtype=torch.int64)), dim=2).view(-1, 2)
        self.weights = torch.ones(self.dim * self.precision, dtype=self.torch_type, device=self.device)
        self.weights /= torch.sum(self.weights)
        self.param_vec = None # Belongs to R^n, or can be constrained to belong to (R^+)^n or [0, 1]^n without loss of expressivity
        self.bit_order_inv_perm = None
        self.param_sanitation_reorder = None
        self.box_param_vec = None
        self.param_vec_grad = None
        self.regularization = torch.linspace(1e-7, 0, self.weights.shape[0], dtype=self.torch_type, device=self.device)
        self.data_bit_array = None
        self.data_reduced = torch.empty((self.total_n, self.dim*self.precision), device=self.device, dtype=torch.int64)
        self.u = None
        self.cu = None
        self.log_det = None
        self.d_logdet_d_weights = None
        self.woodbury = None
        self.woodbury_extended = None
        self.nll = None
        self.dnll_dweights = None
        self.pred_cov_u = None
        self.pred_cov_cu = None
        self.pred_var = None
        self.test_nll = None
        self.obj_id = None
        self.bit_order_for_latest_nll = None
        self.weights_for_latest_nll = None
        self.bit_order_for_latest_grad = None
        self.weights_for_latest_grad = None
        self.make_data_bit_array()

    def _sanitize_bit_order(self):
        """
        ensure that bit (i, j) precedes bit (i+1, j) in self.bit_order
        """
        _, sort_by_dim = torch.sort(self.bit_order[:, 1], stable=True)
        temp_mesh = torch.stack(torch.meshgrid(torch.arange(self.dim, device=self.device), torch.arange(self.precision, device=self.device)), dim=2).view(-1, 2).flip(1)
        self.bit_order[sort_by_dim] = temp_mesh

    def change_active_bit_slice(self, step=1, end=None):
        if end is None:
            end = self.dim * self.precision
        if step > 1:
            end = end - (end % step)
        start = step - 1
        last_weight = self.weights[self.active_bit_slice][-1].clone()
        self.weights[self.active_bit_slice][-1] = 0
        new_weights = torch.sum(self.weights[:end].view(-1, step), dim=1)
        new_weights[-1] = last_weight
        self.active_bit_slice = slice(start, end, step)
        self.data_reduced = torch.empty((self.total_n, end // step), device=self.device, dtype=torch.int64)
        self.data_reduced_vectorizable = None
        self.make_data_reduced()
        self.weights.fill_(0)
        self.weights[self.active_bit_slice] = new_weights
        self.weights[self.active_bit_slice] /= torch.sum(self.weights[self.active_bit_slice])
        self.param_vec = None
        self.param_sanitation_reorder = None
        self.box_param_vec = None
        self.param_vec_grad = None
        # We'll want all these in totally different memory locations on the gpu,
        # and we'll have to recalculate them anyway:
        self.u = None
        self.cu = None
        self.log_det = None
        self.d_logdet_d_weights = None
        self.woodbury = None
        self.woodbury_extended = None
        self.nll = None
        self.dnll_dweights = None
        self.pred_cov_u = None
        self.pred_cov_cu = None
        self.pred_var = None
        self.test_nll = None
        self.bit_order_for_latest_nll = None
        self.weights_for_latest_nll = None
        self.bit_order_for_latest_grad = None
        self.weights_for_latest_grad = None

    def random_bit_order(self):
        self.bit_order = self.bit_order[torch.randperm(self.bit_order.size()[0], device=self.device)]
        self._sanitize_bit_order()
        self.bit_order_inv_perm = bit_order_to_inv_permutation(self.bit_order, self.precision)

    def make_data_bit_array(self):
        if self.precision < 16: # one would hope
            data_int = (self.data * 2**self.precision).to(dtype=torch.int16)
        else:
            data_int = (self.data * 2**self.precision).to(dtype=torch.int64)
        self.data_bit_array = torch.transpose(binary(data_int, self.precision), 1, 2).reshape(self.total_n, -1)

    def make_data_reduced(self):
        """
        The array shuffled_data shuffles the columns of self.data_bit_array according
        to self.bit_order.

        self.data_reduced is an array of integers such that self.data_reduced[i, k] ==
        self.data_reduced[j, k] iff shuffled_data[i, :k+1] == shuffled_data[j, :k+1],
        and max(self.data_reduced) < shuffled_data.shape[0].
        These are the only relevant properties of the array.
        """
        self.data_reduced[:, -1] = torch.arange(self.total_n, dtype=self.data_reduced.dtype, device=self.device)
        first_irrelevant_bit = self.active_bit_slice.stop - self.active_bit_slice.step
        shuffled_data = self.data_bit_array[:, self.bit_order[:first_irrelevant_bit, 0]*self.dim + self.bit_order[:first_irrelevant_bit, 1]]
        shuffled_data_byte_form = bit_array_to_byte_array(shuffled_data)
        tmp, last_col_reduced = torch.unique(shuffled_data_byte_form, dim=0, sorted=True, return_inverse=True)
        unique_data = torch.empty_like(shuffled_data[:tmp.shape[0]])
        unique_data[last_col_reduced] = shuffled_data # these last few rows make torch.unique faster by storing 8 bits in one byte rather than 8 bytes.

        # unique_data_good, last_col_reduced_good = torch.unique(shuffled_data, dim=0, sorted=True, return_inverse=True)
        # assert(torch.equal(unique_data, unique_data_good))

        # what follows relies on undocumented behavior that unique_data will be sorted lexically by first column, then second column, and so on
        self.data_reduced[:, -2] = last_col_reduced
        diff_from_above = torch.empty_like(unique_data[:, :-1])
        diff_from_above[0] = 0 * diff_from_above[0]
        diff_from_above[1:] = torch.abs(torch.sign(torch.diff(unique_data[:, :-1], dim=0)))
        # diff_from_above[i, j] is true if unique_data[i, j] == unique_data[i-1, j]
        diff_from_above = torch.sign(torch.cumsum(diff_from_above, dim=1))
        # now diff_from_above[i, j] is true if unique_data[i, k] == unique_data[i-1, k] for all k <= j
        data_reduced_sorted = torch.cumsum(diff_from_above.to(dtype=torch.int64), 0)
        # data_reduced_sorted[i, j] is the number of times diff_from_above[k, j] for k <= i is True,
        # so (data_reduced_sorted[i, j] == data_reduced_sorted[k, j]) implies
        # data_reduced_sorted[i, :j+1] == data_reduced_sorted[k, :j+1]
        self.data_reduced[:, :-2] = data_reduced_sorted[:, self.active_bit_slice][last_col_reduced]
        self.data_reduced_vectorizable = self.data_reduced + self.data_reduced.shape[0] * torch.arange(self.data_reduced.shape[1], device=self.device).view(1, -1)

    def invert(self, different_lamb=None):
        if different_lamb is None:
            lamb = self.lamb
        else:
            lamb = different_lamb
            self.lamb = lamb
        self.u = torch.ones((self.train_n, self.data_reduced.shape[1]), dtype=self.weights.dtype, device=self.data_reduced.device)
        self.cu = self.u.clone()
        self.log_det = math.log(lamb) * self.train_n
        for col in range(self.data_reduced.size()[1] - 1, -1, -1):
            dot_products = torch.zeros(self.total_n, dtype=self.weights.dtype, device=self.data_reduced.device)
            dot_products.index_add_(0, self.data_reduced[:self.train_n, col], self.u[:, col]) # implicitly u[:, col] * torch.ones_like(u[:, col])
            expanded_dot_products = dot_products[self.data_reduced[:self.train_n, col]]
            self.cu[:, col] = -self.weights[self.active_bit_slice][col] / (lamb + self.weights[self.active_bit_slice][col] * expanded_dot_products) * self.u[:, col]
            if col > 0:
                self.u[:, col-1] = self.u[:, col] + self.cu[:, col] * expanded_dot_products
            self.log_det += torch.sum(torch.log(1 + dot_products * self.weights[self.active_bit_slice][col] / lamb))

    def calculate_woodbury(self):
        # self.woodbury = (matvec(self.data_reduced[:self.train_n], self.u, self.cu, self.targets, extra_indices=self.total_n) + self.targets)/self.lamb
        self.woodbury = (matvec_vectorizable(self.data_reduced_vectorizable[:self.train_n], self.u, self.cu, self.targets, extra_indices=self.total_n) + self.targets)/self.lamb
        self.woodbury_extended = torch.cat((self.woodbury, torch.zeros(self.total_n-self.train_n, device=self.woodbury.device, dtype=self.woodbury.dtype)))

    def calculate_neg_log_likelihood(self):
        self.nll = (self.targets @ self.woodbury + self.log_det + self.train_n * math.log(2 * math.pi)) / 2 / self.train_n
        self.bit_order_for_latest_nll = self.bit_order.clone()
        self.weights_for_latest_nll = self.weights.clone()
        return self.nll

    def calculate_trace_Kinv_dK_dweights(self):
        self.d_logdet_d_weights = torch.zeros_like(self.weights[self.active_bit_slice])
        u_sums = torch.zeros((self.total_n, self.data_reduced.shape[1]), dtype=self.u.dtype, device=self.device)
        cu_sums = torch.zeros((self.total_n, self.data_reduced.shape[1]), dtype=self.cu.dtype, device=self.device)
        prod = torch.zeros_like(u_sums)
        for col in range(self.data_reduced.shape[1]):
            u_sums[:, :col].fill_(0) # other columns already 0
            cu_sums[:, :col].fill_(0)
            prod[:, :col].fill_(0)
            u_sums[:, :col+1].index_add_(0, self.data_reduced[:self.train_n, col], self.u[:, :col+1]) # a lot of these sums were just done in the last iteration of the loop; EDIT: And yet... redoing them is quicker than applying the boolean mask necessary to do computations on select rows
            cu_sums[:, :col+1].index_add_(0, self.data_reduced[:self.train_n, col], self.cu[:, :col+1]) # these are just a constant times the sums above
            prod[:, :col+1] = u_sums[:, :col+1] * cu_sums[:, :col+1]
            sums = torch.sum(prod[:, :col+1], dim=0)
            self.d_logdet_d_weights[col] += torch.sum(sums)
            self.d_logdet_d_weights[:col] += sums[col]
        self.d_logdet_d_weights += self.train_n # from the contribution of the identity matrix within K_inv
        self.d_logdet_d_weights /= self.lamb
        return self.d_logdet_d_weights

    def calculate_dnll_dweights(self, regularization=False):
        if self.bit_order_for_latest_grad is not None:
            if torch.all(self.bit_order == self.bit_order_for_latest_grad):
                if torch.all(self.weights == self.weights_for_latest_grad):
                    return self.dnll_dweights
        self.calculate_trace_Kinv_dK_dweights()
        dot_products = torch.zeros(self.total_n, dtype=self.woodbury.dtype, device=self.device)
        wood_dKdw_wood = torch.zeros_like(self.d_logdet_d_weights)
        for col in range(self.data_reduced.size()[1]):
            dot_products.fill_(0)
            dot_products.index_add_(0, self.data_reduced[:self.train_n, col], self.woodbury)
            wood_dKdw_wood[col] = dot_products @ dot_products
        self.dnll_dweights = (-wood_dKdw_wood + self.d_logdet_d_weights) / (2 * self.train_n)
        if regularization:
            self.dnll_dweights -= self.regularization[self.active_bit_slice]
        self.bit_order_for_latest_grad = self.bit_order.clone()
        self.weights_for_latest_grad = self.weights.clone()
        return self.dnll_dweights

    def process(self, data_already_reduced=False):
        if self.data_bit_array is None:
            self.make_data_bit_array()
        if self.bit_order_for_latest_nll is not None:
            if torch.all(self.bit_order == self.bit_order_for_latest_nll):
                data_already_reduced = True
                if torch.all(self.weights == self.weights_for_latest_nll):
                    return
        if not data_already_reduced:
            self.make_data_reduced()
        self.invert()
        self.calculate_woodbury()
        self.calculate_neg_log_likelihood()

    def constrain_weights(self, min_weight=None):
        if min_weight is None:
            min_weight = self.min_weight
        self.weights[:] = torch.clamp(self.weights, min=min_weight)
        self.weights[self.active_bit_slice] /= torch.sum(self.weights[self.active_bit_slice])

    def bit_order_and_weights_to_param_vec(self):
        # param_vec_old = boxize_old(bit_order_to_permutation(self.bit_order, self.precision), self.weights)
        # self.param_vec = boxize(bit_order_to_inv_permutation(self.bit_order, self.precision), self.weights)
        self.param_vec = boxize(self.bit_order_inv_perm, self.weights)
        # assert(torch.equal(param_vec_old, self.param_vec))
        return self.param_vec

    def param_vec_to_bit_order_and_weights(self):
        # param_vec_san, self.param_sanitation_reorder = sanitize_vec(rn_to_box(self.param_vec), self.precision)
        # self.box_param_vec = rn_to_box(param_vec_san)
        self.box_param_vec = rn_to_box(self.param_vec)
        perm, self.weights = vec_to_perm_and_weights(self.box_param_vec)
        self.bit_order = permutation_to_bit_order(perm, self.precision)
        self.bit_order_inv_perm = torch.argsort(perm)
        return self.bit_order, self.weights

    def calculate_dnll_dparamvec(self, c=1.):
        self.calculate_dnll_dweights()
        # box_vec_grad_old = grad_to_vec_grad_old(bit_order_to_permutation(self.bit_order, self.precision), self.dnll_dweights)
        # box_vec_grad = grad_to_vec_grad(bit_order_to_inv_permutation(self.bit_order, self.precision), self.dnll_dweights)
        box_vec_grad = grad_to_vec_grad(self.bit_order_inv_perm, self.dnll_dweights)
        # assert(torch.equal(box_vec_grad_old, box_vec_grad))
        vec_grad_san = box_vec_grad_to_rn_vec_grad(box_vec_grad, self.param_vec, self.box_param_vec, c=c)
        # self.param_vec_grad = vec_grad_sanitized_to_unsanitized(vec_grad_san, self.param_sanitation_reorder)
        self.param_vec_grad = vec_grad_san
        return self.param_vec_grad

    def opt_weights_and_bit_order_bfgs(self, max_iter=1000, tol=1e-4, verbose=False, debug=False):
        assert(self.active_bit_slice.step == 1)
        assert(self.active_bit_slice.stop == self.dim * self.precision)
        x = torch.log(self.bit_order_and_weights_to_param_vec())
        def get_loss():
            self.param_vec = torch.exp(x - torch.max(x))
            self.param_vec_to_bit_order_and_weights()
            self.process()
            loss = self.nll + extra_loss(self.param_vec, c=1.)
            if debug:
                assert(torch.isfinite(loss))
            return loss
        def get_grad():
            self.param_vec = torch.exp(x - torch.max(x))
            self.param_vec_to_bit_order_and_weights()
            self.process()
            return self.calculate_dnll_dparamvec(c=1.) * self.param_vec
        def get_loss_and_grad():
            self.param_vec = torch.exp(x - torch.max(x))
            self.param_vec_to_bit_order_and_weights()
            self.process()
            loss = self.nll + extra_loss(self.param_vec, c=1.)
            grad = self.calculate_dnll_dparamvec(c=1.) * self.param_vec
            return loss, grad
        def print_loss():
            loss = self.nll + extra_loss(self.param_vec, c=1.)
            print("loss:", loss.item())
        bfgs = BFGS(x, get_loss, get_grad, get_loss_and_grad)
        bfgs.minimize(max_iter=max_iter, tol=tol, callable=print_loss if verbose else None, verbose=verbose, debug=debug)

    def scipy_opt_weights_and_bit_order_bfgs(self, max_iter=1000, verbose=True):
        assert(self.active_bit_slice.step == 1)
        assert(self.active_bit_slice.stop == self.dim * self.precision)
        x = torch.log(self.bit_order_and_weights_to_param_vec()).clone().cpu().numpy()

        def scipy_loss(x):
            self.param_vec = torch.exp(torch.from_numpy(x).view(-1).to(dtype=self.torch_type, device=self.device))
            self.param_vec_to_bit_order_and_weights()
            self.process()
            loss_gpu = self.nll + extra_loss(self.param_vec, c=1.)
            return loss_gpu.item()

        def scipy_grad(x):
            self.param_vec = torch.exp(torch.from_numpy(x).view(-1).to(dtype=self.torch_type, device=self.device))
            self.param_vec_to_bit_order_and_weights()
            self.process()
            grad = self.calculate_dnll_dparamvec(c=1.)
            return (grad * self.param_vec).cpu().numpy()

        def between_iterations(x):
            if verbose:
                print("Train nll:", self.nll)

        res = minimize(
                    scipy_loss,
                    x,
                    method="BFGS",
                    jac=scipy_grad,
                    options={"maxiter": max_iter, "disp": True},
                    callback=between_iterations
                )
        if not res.success:
            try:
                # Some res.message are bytes
                msg = res.message.decode("ascii")
            except AttributeError:
                # Others are str
                msg = res.message
            print(f"Fitting failed with the optimizer reporting '{msg}'")
        self.param_vec = torch.exp(torch.from_numpy(res.x).view(-1).to(dtype=self.torch_type, device=self.device))
        self.param_vec_to_bit_order_and_weights()
        self.process()

    def predict_test_means(self):
        # K_wood_extended = matvec_with_just_weights(self.data_reduced, self.weights[self.active_bit_slice], self.woodbury_extended)
        K_wood_extended = matvec_with_just_weights_vectorizable(self.data_reduced_vectorizable, self.weights[self.active_bit_slice], self.woodbury_extended)
        return K_wood_extended[self.train_n:]

    def calculate_predictive_precision(self):
        # this uses the fact that a schur complement is a block of the relevant inverse matrix
        lamb = self.lamb
        u_columns = [torch.ones(self.total_n, dtype=self.weights.dtype, device=self.data_reduced.device)]
        cs_columns = []
        for col in range(self.data_reduced.size()[1] - 1, -1, -1):
            dot_products = torch.zeros(self.total_n, dtype=self.weights.dtype, device=self.data_reduced.device)
            dot_products.index_add_(0, self.data_reduced[:, col], u_columns[-1]) # implicitly u[:, col] * torch.ones_like(u[:, col])\
            dot_proudcts_extended = dot_products[self.data_reduced[:, col]]
            cs_columns.append(-self.weights[self.active_bit_slice][col] / (lamb + self.weights[self.active_bit_slice][col] * dot_proudcts_extended))
            u_columns.append(u_columns[-1] * (1 + cs_columns[-1] * dot_proudcts_extended))
        u_columns = [col[self.train_n:] for col in u_columns]
        cs_columns = [col[self.train_n:] for col in cs_columns]
        self.pred_prec_cs = torch.stack(cs_columns[::-1], dim=1) # Now that you're not using autograd; allocate the memory for the whole array first and slot it in, rather than copying it over
        self.pred_prec_u = torch.stack(u_columns[:-1][::-1], dim=1)
        # add the identity matrix, then divide everything by lamb to get the precision matrix

    def calculate_predictive_variance(self):
        self.pred_cov_u = self.pred_prec_u.clone()
        self.pred_cov_cu = torch.empty_like(self.pred_cov_u)
        dot_products = torch.empty(self.total_n, dtype=self.weights.dtype, device=self.data_reduced.device)
        dot_products_2 = torch.empty((self.total_n, self.pred_cov_u.size()[1]-1), dtype=self.weights.dtype, device=self.data_reduced.device)
        for col in range(self.data_reduced.size()[1] - 1, -1, -1):
            # dot_products *= 0
            dot_products = torch.zeros(self.total_n, dtype=self.weights.dtype, device=self.data_reduced.device)
            dot_products.index_add_(0, self.data_reduced[self.train_n:, col], self.pred_cov_u[:, col] * self.pred_prec_u[:, col])
            self.pred_cov_cu[:, col] = -self.pred_prec_cs[:, col] / (1 + self.pred_prec_cs[:, col] * dot_products[self.data_reduced[self.train_n:, col]]) * self.pred_cov_u[:, col]
            if col > 0:
                dot_products_2 *= 0
                dot_products_2[:, :col].index_add_(0, self.data_reduced[self.train_n:, col], self.pred_cov_u[:, col:col+1] * self.pred_prec_u[:, :col])
                self.pred_cov_u[:, :col] += self.pred_cov_cu[:, col:col+1] * dot_products_2[self.data_reduced[self.train_n:, col], :col]
        # get the diagonal elements
        self.pred_var = torch.sum(self.pred_cov_cu * self.pred_cov_u, axis=-1) * self.lamb

    def predict_test_var(self):
        self.calculate_predictive_precision()
        self.calculate_predictive_variance()
        return self.pred_var

    def rmse(self, targets):
        return rmse(targets, self.predict_test_means())

    def calculate_test_nll(self, targets):
        errors = targets - self.predict_test_means()
        var = self.predict_test_var()
        std = torch.sqrt(var)
        self.test_nll = torch.mean((torch.square(errors / std) + math.log(2 * math.pi)) / 2 + torch.log(std))
        return self.test_nll

    def calculate_predictive_variance_once(self, test_index):
        one_hot = torch.zeros(self.total_n, dtype=self.weights.dtype, device=dev)
        one_hot[test_index + self.train_n] = 1
        kernel_vector = matvec_with_just_weights(self.data_reduced, self.weights[self.active_bit_slice], one_hot)[:self.train_n]
        return 1 - kernel_vector @ (matvec(self.data_reduced[:self.train_n], self.u, self.cu, kernel_vector, extra_indices=self.total_n) + kernel_vector)/self.lamb

    def to(self, device):
        for key in self.__dict__:
            if self.__dict__[key] is not None:
                if torch.is_tensor(self.__dict__[key]):
                    setattr(self, key, self.__dict__[key].to(device=device))

    def save(self, path="/content/gdrive/MyDrive/ongp_results/"):
        if self.obj_id is None:
            self.obj_id = str(torch.randint(high=10**8, size=(1,)).item())
        torch.save(self.weights.to(device='cpu'), path + self.obj_id + "_weights.pt")
        torch.save(self.bit_order.to(device='cpu'), path + self.obj_id + "_bit_order.pt")

    def load(self, path="/content/gdrive/MyDrive/ongp_results/"):
        self.weights = torch.load(path + self.obj_id + "_weights.pt").to(device=dev)
        self.bit_order = torch.load(path + self.obj_id + "_bit_order.pt").to(device=dev)
