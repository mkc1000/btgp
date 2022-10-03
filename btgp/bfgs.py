import torch
import time

def linesearch(params, grad, dir, get_loss, lr=1., init_loss=None, c=0.1, tol=1e-15, jump_factor=2, verbose=False, debug=True):
    """
    params is modified in place
    get_loss should take no arguments, but should change its output
    when the variable params is changed
    outputs new loss and latest learning rate
    if latest learning rate is None, we are already at minimum in that direction (within machine precision)
    """
    init_loss = None # don't know what's gone wrong, but this should fix it cheaply
    if init_loss is None:
        init_loss = get_loss()
    if verbose:
        print("init_loss", init_loss)
    params_orig = params.clone()
    params += lr * dir
    if (dir/torch.norm(dir)) @ (-grad/torch.norm(grad)) <= 0: # divide by norm for underflow reasons
        print("WARNING -dir/|dir| @ grad/|grad| =", (dir/torch.norm(dir)) @ (-grad/torch.norm(grad)))
        dir = -lr * grad
    new_loss = get_loss()
    if verbose:
        print("new_loss", new_loss)
        print("init_loss - new_loss", init_loss - new_loss)
    if init_loss - new_loss >=  c * (-lr * dir @ grad):
        if verbose:
            print("increasing lr")
        # keep going!
        latest_loss = new_loss.clone()
        while new_loss <= latest_loss:
            latest_loss = new_loss.clone()
            lr *= jump_factor
            params[:] = params_orig + lr * dir
            new_loss = get_loss()
        lr /= jump_factor
        params[:] = params_orig + lr * dir
        if verbose:
            print("final_loss", get_loss())
        if debug:
            assert(get_loss() <= init_loss)
        return get_loss(), lr
    else:
        # backtrack
        while init_loss - new_loss < c * (-lr * dir @ grad):
            lr /= jump_factor
            if verbose:
                print("init_loss - new_loss", init_loss - new_loss)
                print("lr", lr)
            if c * (-lr * dir @ grad) < tol:
                if verbose:
                    print("-dir @ grad", -dir @ grad)
                    print("(-dir/||dir|| @ grad/||grad||)", -(dir/torch.norm(dir)) @ grad/torch.norm(grad))
                    print("No improvement possible")
                params[:] = params_orig
                if verbose:
                    print("final_loss", get_loss())
                loss = get_loss()
                return get_loss(), None
            params[:] = params_orig + lr * dir
            new_loss = get_loss()
            if verbose:
                print("new_loss", new_loss)
        if verbose:
            print("final_loss", get_loss())
        if debug:
            assert(torch.all(torch.isfinite(params)))
        if new_loss > init_loss:
            print("WARNING: Loss increased by", new_loss - init_loss)
        return new_loss, lr
      
class BFGS(object):
    def __init__(self, params, get_loss, get_grad, get_loss_and_grad, init_hess_approx=1.0, init_lr=1.0):
        """
        params is modified in place
        get_loss, get_grad, get_loss_and_grad should take no arguments, but
        should change their output when the variable params is changed
        """
        self.H_inv = torch.eye(params.shape[0], dtype=params.dtype, device=params.device) / init_hess_approx
        self.lr = init_lr
        self.params = params
        self.get_loss = get_loss
        self.get_grad = get_grad
        self.get_loss_and_grad = get_loss_and_grad
        self.loss, self.grad = get_loss_and_grad()
        self.n_steps = 0

    def update(self, recompute_init_loss_and_grad=False, tol=1e-6, verbose=False, debug=True):
        """Returns True if minimum achieved; else False"""
        if recompute_init_loss_and_grad:
            self.loss, self.grad = self.get_loss_and_grad()
        dir = -self.H_inv @ self.grad
        if debug:
            assert(torch.all(torch.isfinite(self.grad)))
            assert(torch.all(torch.isfinite(dir)))
        old_params = self.params.clone()
        self.loss, lr = linesearch(self.params, self.grad, dir, self.get_loss,
                                   lr=self.lr, init_loss=self.loss, tol=tol, verbose=verbose, debug=debug)
        if lr is not None:
            self.lr = lr
        else: # lr is None; linesearch failed to improve loss
            # try again going straight opposite the direction of grad
            self.loss, lr = linesearch(self.params, self.grad, -self.grad.clone(), self.get_loss,
                                       lr=self.lr, init_loss=self.loss, tol=tol, verbose=verbose, debug=debug)
            if lr is not None:
                # If it worked this time, the Hessian was misleading us
                if verbose:
                    print("WARNING: Hessian became unstable; reinitializing")
                self.lr = 1.0
                self.H_inv = torch.eye(self.params.shape[0], dtype=self.params.dtype, device=self.params.device) * lr
            else:
                return True
        if verbose:
            print("lr :", self.lr)
        s = self.lr * dir
        new_grad = self.get_grad()
        y = new_grad - self.grad
        curv = s @ y
        self.grad = new_grad
        if curv <= 0:
            if verbose:
                print("WARNING: negative curvature detected; skipping BFGS update")
            # y += (-curv + 1e-4) / (s @ s) * s
            # curv = 1e-4
            # self.H_inv = torch.eye(self.H_inv.shape[0], dtype=self.H_inv.dtype, device=self.H_inv.device) * self.lr
            # self.lr = 1.0
        else:
            # M = -torch.outer(s, y)/curv + torch.eye(s.shape[0], device=s.device, dtype=s.dtype)
            # self.H_inv = (M @ self.H_inv) @ M.t() + torch.outer(s, s)/curv
            H_inv_y = self.H_inv @ y
            M = torch.outer(s, H_inv_y) / curv
            self.H_inv += -(M + M.t())
            self.H_inv += (curv + H_inv_y @ y)/(curv**2) * torch.outer(s, s)
            self.n_steps += 1
        return False

    def minimize(self,  max_iter=1000, max_seconds=None, tol=1e-6, callable=None, verbose=False, debug=True):
        if max_seconds is not None:
            start = time.time()
        for _ in range(max_iter):
            if callable is not None:
                callable()
            if max_seconds is not None:
                if time.time() - start > max_seconds:
                    print("Alloted optimization time exceeded")
                    break
            if self.update(tol=tol, verbose=verbose, debug=debug):
                break
