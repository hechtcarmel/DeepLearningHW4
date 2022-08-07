import torch

from attacks.optimizers.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, a_abs, multiplier):
        self.a_abs = a_abs
        self.multiplier = multiplier

    def calc_step(self, w, dw=None):
        with torch.no_grad():
            dw = w.grad if dw is None else dw
            step = self.a_abs * self.multiplier * dw
        return step
