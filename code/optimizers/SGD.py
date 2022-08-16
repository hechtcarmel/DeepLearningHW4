import torch

from optimizers.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, args):
        self.a_abs = getattr(args, 'a_abs', 1)
        self.multiplier = getattr(args, 'multiplier', 1)
        super().__init__()

    def calc_step(self, w, dw=None):
        with torch.no_grad():
            dw = w.grad if dw is None else dw
            step = self.a_abs * self.multiplier * dw
        return step
