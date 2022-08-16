import torch

from optimizers.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, a_abs=1, multiplier=1, **kwargs):
        self.a_abs = a_abs
        self.multiplier = multiplier
        super().__init__()

    def calc_step(self, w, dw=None):
        with torch.no_grad():
            dw = w.grad if dw is None else dw
            step = self.a_abs * self.multiplier * dw
        return step
