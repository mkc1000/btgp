import numpy as np
import torch
from btgp.gp import ONGP
import os


class GPInfo(object):
    def __init__(self, gp, obj_id=None):
        self.weights = gp.weights.clone()
        self.bit_order = gp.bit_order.clone()
        self.nll = gp.nll
        self.lr = gp.lr
        self.obj_id = obj_id

    def modify_gp(self, gp):
        gp.weights = self.weights.clone()
        gp.bit_order = self.bit_order.clone()
        gp.lr = self.lr

    def save(self, path="ongp_results"):
        if self.obj_id is None:
            self.obj_id = str(torch.randint(high=10 ** 8, size=(1,)).item())

        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        torch.save(
            self.weights.cpu(), os.path.join(output_dir, f"{self.obj_id}_weights.pt")
        )
        torch.save(
            self.bit_order.cpu(),
            os.path.join(output_dir, f"{self.obj_id}_bit_order.pt"),
        )
        torch.save(
            torch.tensor(self.lr).to(device="cpu"),
            os.path.join(output_dir, f"{self.obj_id}_lr.pt"),
        )

    def load(
        self,
        path="/content/gdrive/MyDrive/ongp_results/",
        device="cpu",
        new_obj_id=None,
    ):
        if new_obj_id is not None:
            self.obj_id = new_obj_id
        self.weights = torch.load(path + self.obj_id + "_weights.pt").to(device=device)
        self.bit_order = torch.load(path + self.obj_id + "_bit_order.pt").to(
            device=device
        )
        self.lr = torch.load(path + self.obj_id + "_lr.pt").item()


class GPPopulation(object):
    def __init__(
        self,
        train_data,
        targets,
        test_data,
        pop_size=10,
        precision=8,
        lambd=0.0001,
        default_lr=0.01,
        eps=0.01,
        quickstart=False,
    ):
        self.pop_size = pop_size
        self.train_data = train_data
        self.targets = targets
        self.test_data = test_data
        self.precision = precision
        self.lamb = lambd
        self.lr = default_lr
        self.eps = eps
        self.gp = ONGP(
            self.train_data,
            self.targets,
            self.test_data,
            precision=self.precision,
            lambd=self.lamb,
            default_lr=self.lr,
            eps=self.eps,
        )
        self.gps = []
        self.n_gradient_steps = []
        self.rejects = []
        for _ in range(self.pop_size):
            if quickstart:
                self.quick_add_new_gp()
            else:
                self.add_new_gp()
        self.nlls = None

    def quick_add_new_gp(self):
        self.gp.random_bit_order()
        self.gps.append(GPInfo(self.gp))
        self.n_gradient_steps.append(0)

    def add_new_gp(self):
        self.gp.random_bit_order()
        self.gp.process()
        for _ in range(5):
            self.gp.update_weights(update_order=False)
        self.gps.append(GPInfo(self.gp))
        self.n_gradient_steps.append(0)

    def get_nlls(self):
        self.nlls = np.array([gp.nll.cpu().numpy() for gp in self.gps])

    def sort_best_to_worst(self):
        self.get_nlls()
        order = np.argsort(self.nlls)
        self.gps = [self.gps[i] for i in order]
        self.n_gradient_steps = [self.n_gradient_steps[i] for i in order]
        self.nlls = self.nlls[order]

    def mate(self, index_a, index_b):
        self.quick_add_new_gp()
        bit_orders = [
            self.gps[index].bit_order[:, :1] * self.gp.dim
            + self.gps[index].bit_order[:, 1:]
            for index in [index_a, index_b]
        ]
        new_bit_order = torch.hstack(bit_orders)
        weights = torch.hstack(
            [self.gps[index].weights.view(-1, 1) for index in [index_a, index_b]]
        ).view(-1)
        new_bit_order = new_bit_order.view(-1)
        sorted, indcs = torch.sort(new_bit_order, stable=True)
        sorted_weights = weights[indcs]
        indcs = indcs.view(-1, 2)[:, 0]
        sorted = sorted.view(-1, 2)[:, 0]
        sorted_weights = sorted_weights.view(-1, 2)[:, 0]
        new_bit_order = -1 * torch.ones_like(new_bit_order)
        new_bit_order[indcs] = sorted
        new_bit_order = new_bit_order[new_bit_order != -1]
        new_weights = -1 * torch.ones_like(weights)
        new_weights[indcs] = sorted_weights
        new_weights = new_weights[new_weights != -1]
        new_weights /= torch.sum(new_weights)
        self.gps[-1].bit_order = torch.empty_like(self.gps[index_a].bit_order)
        self.gps[-1].bit_order[:, 1] = torch.fmod(new_bit_order, self.gp.dim)
        self.gps[-1].bit_order[:, 0] = torch.div(
            new_bit_order, self.gp.dim, rounding_mode="floor"
        )
        self.gps[-1].weights = new_weights

    def mutate(self, index, n_flips=1, copy=False):
        if copy:
            self.quick_add_new_gp()
            self.gps[-1].bit_order = self.gps[index].bit_order.clone()
            self.gps[-1].weights = self.gps[index].weights.clone()
            index = -1
        for _ in range(n_flips):
            location = torch.randint(0, self.gps[index].bit_order.shape[0] - 1, 1)
            tmp = self.gps[index].bit_order[location].clone()
            self.gps[index].bit_order[location] = self.gps[index].bit_order[
                location + 1
            ]
            self.gps[index].bit_order[location + 1] = tmp
        self.gps[index].bit_order = sanitize_bit_order(
            self.gps[index].bit_order, dim=self.gp.dim, precision=self.gp.precision
        )

    def cull_bottom(self, quantile=0.2):
        self.sort_best_to_worst()
        keep = len(self.gps) - int(quantile * len(self.gps))
        self.rejects += self.gps[keep:]
        self.gps = self.gps[:keep]
        self.n_gradient_steps = self.n_gradient_steps[:keep]
        self.nlls = self.nlls[:keep]

    def process_index(self, index):
        self.gps[index].modify_gp(self.gp)
        self.gp.process()
        self.gps[index] = GPInfo(self.gp, obj_id=self.gps[index].obj_id)

    def process_all(self):
        for idx in range(len(self.gps)):
            self.process_index(idx)

    def improve_index(self, index, n_times=10):
        self.gps[index].modify_gp(self.gp)
        self.gp.process()
        for _ in range(n_times):
            self.gp.update_weights(update_order=True)
        self.gps[index] = GPInfo(self.gp)
        self.n_gradient_steps[index] += n_times

    def improve_all(self, n_times=10):
        for i in range(len(self.gps)):
            self.improve_index(i, n_times=n_times)

    def order_similarity(self):
        similarity_matrix = np.zeros((len(self.gps), len(self.gps)))
        for i in range(len(self.gps)):
            for j in range(i + 1):
                similarity_matrix[i, j] = (
                    torch.sum(
                        self.gps[i].bit_order[:, 1] == self.gps[j].bit_order[:, 1]
                    )
                    .cpu()
                    .item()
                )
                similarity_matrix[j, i] = similarity_matrix[i, j]
        return similarity_matrix / self.gps[i].bit_order.shape[0]

    def col_better_than_row(self):
        return pairwise_comparison(self.nlls).T

    def best_of_its_kind(self, similarity_threshold=0.8):
        col_dominates_row = np.logical_and(
            self.col_better_than_row(), (self.order_similarity() > similarity_threshold)
        )
        dominated = np.any(col_dominates_row, axis=1)
        return np.logical_not(dominated)

    def save(self):
        for gp in self.gps:
            gp.save()
        for gp in self.rejects:
            gp.save()


def pairwise_comparison(vec, tol=0):
    n = vec.shape[0]
    out = np.zeros((n, n), dtype=np.bool)
    for i in range(n):
        out[i] = vec[i] < vec - tol
        out[i, i] = False
    return out


def sanitize_bit_order(order, dim, precision):
    order_ = order.clone()
    _, sort_by_dim = torch.sort(order_[:, 1], stable=True)
    temp_mesh = (
        torch.stack(
            torch.meshgrid(
                torch.arange(dim, device=order.device),
                torch.arange(precision, device=order.device),
            ),
            dim=2,
        )
        .view(-1, 2)
        .flip(1)
    )
    order_[sort_by_dim] = temp_mesh
    return order_
