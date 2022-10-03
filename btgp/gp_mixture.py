import numpy as np
import torch
from efficient_gp.gp import ONGP
import os
import math

class GPInfo(object):
    def __init__(self, gp, obj_id=None):
        self.weights = gp.weights.clone()
        self.bit_order = gp.bit_order.clone()
        self.nll = gp.nll.clone()
        self.obj_id = obj_id

    def modify_gp(self, gp):
        gp.weights = self.weights.clone()
        gp.bit_order = self.bit_order.clone()

    def save(self, path="ongp_results"):
        if self.obj_id is None:
            self.obj_id = str(torch.randint(high=10**8, size=(1,)).item())
        torch.save(self.weights.to(device='cpu'), path + self.obj_id + "_weights.pt")
        torch.save(self.bit_order.to(device='cpu'), path + self.obj_id + "_bit_order.pt")

    def load(self, path="ongp_results", device='cpu', new_obj_id=None):
        if new_obj_id is not None:
            self.obj_id = new_obj_id
        self.weights = torch.load(path + self.obj_id + "_weights.pt").to(device=device)
        self.bit_order = torch.load(path + self.obj_id + "_bit_order.pt").to(device=device)

class GPMixture(object):
    def __init__(self, train_data, targets, test_data, precision='auto',
                 lambd=0.0001, min_weight=0.01):
        self.train_data = train_data
        self.targets = targets
        self.test_data = test_data
        if precision == 'auto':
            self.precision = (150 // self.train_data.shape[1]) + 1
            if self.precision > 8:
                self.precision = 8
        else:
            self.precision = precision
        self.lamb = lambd
        self.min_weight = min_weight
        self.gp = ONGP(self.train_data, self.targets, self.test_data,
                       precision=self.precision, lambd=self.lamb,
                       min_weight=self.min_weight)
        self.gps = []
        self.rejects = []
        self.nlls = None
        self.mixture_nll = None

    def add(self, gp):
        self.gps.append(GPInfo(gp))
    
    def get_nlls(self):
        self.nlls = np.array([gp.nll.cpu().numpy() for gp in self.gps])
        return self.nlls

    def sort_best_to_worst(self):
        self.get_nlls()
        order = np.argsort(self.nlls)
        self.gps = [self.gps[i] for i in order]
        self.nlls = self.nlls[order]

    def cull_bottom(self, quantile=0.2):
        self.sort_best_to_worst()
        keep = len(self.gps) - int(quantile * len(self.gps))
        self.rejects += self.gps[keep:]
        self.gps = self.gps[:keep]
        self.nlls = self.nlls[:keep]

    def process_index(self, index):
        self.gps[index].modify_gp(self.gp)
        self.gp.process()
        self.gps[index] = GPInfo(self.gp, obj_id=self.gps[index].obj_id)
    
    def process_all(self):
        for idx in range(len(self.gps)):
            self.process_index(idx)
            
    def get_test_nlls(self, test_y):
        self.test_nlls = torch.empty(size=(len(self.gps),), dtype=self.gp.weights.dtype, device=self.gp.device)
        for idx in range(len(self.gps)):
            self.gps[idx].modify_gp(self.gp)
            self.gp.process()
            self.test_nlls[idx] = self.gp.calculate_test_nll(test_y)
        return self.test_nlls

    def gp_mixture_nll(self, test_y, weight_func=(lambda nlls: torch.ones_like(nlls))):
        marg_probs = torch.zeros_like(test_y)
        torch_nlls = torch.tensor(self.nlls).to(device=self.gp.device, dtype=self.gp.weights.dtype)
        torch_nlls -= torch.min(torch_nlls)
        weights = weight_func(torch_nlls)
        weights /= torch.sum(weights)
        for idx in range(len(self.gps)):
            self.gps[idx].modify_gp(self.gp)
            self.gp.process()
            errors = test_y - self.gp.predict_test_means()
            vars = self.gp.predict_test_var()
            stds = torch.sqrt(vars)
            marg_probs += weights[idx] * torch.exp(-((torch.square(errors / stds) + math.log(2 * math.pi)) / 2 + torch.log(stds)))
        self.mixture_nll = torch.mean(-torch.log(marg_probs))
        return self.mixture_nll
    
    def gp_mixture_rmse(self, test_y, weight_func=(lambda nlls: torch.ones_like(nlls))):
        rmses = torch.zeros_like(test_y)
        torch_nlls = torch.tensor(self.nlls).to(device=self.gp.device, dtype=self.gp.weights.dtype)
        torch_nlls -= torch.min(torch_nlls)
        weights = weight_func(torch_nlls)
        weights /= torch.sum(weights)
        pred_means = []
        for idx in range(len(self.gps)):
            self.gps[idx].modify_gp(self.gp)
            self.gp.process()
            pred_means.append(self.gp.predict_test_means())
        mixture_pred = (torch.stack(pred_means, dim=-1) * weights).sum(dim=-1)
        rmse = (test_y - mixture_pred).pow(2).mean().sqrt()
        return rmse

    def save(self):
        for gp in self.gps:
            gp.save()
        for gp in self.rejects:
            gp.save()
