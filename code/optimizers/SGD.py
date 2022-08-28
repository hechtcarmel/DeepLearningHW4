import torch

from optimizers.Optimizer import Optimizer


class SGD(Optimizer):
    # def __init__(self, args):
    #     self.a_abs = getattr(args, 'a_abs', 1)
    #     self.multiplier = getattr(args, 'multiplier', 1)
    #     super().__init__()

    def __init__(self, a_abs=1, multiplier=1):
        self.a_abs = a_abs
        self.multiplier = multiplier
        super().__init__()


    def calc_step(self, w=None, dw=None):
        with torch.no_grad():
            dw = w.grad if dw is None else dw
            assert dw is not None
            assert self.a_abs is not None
            assert self.multiplier is not None
            step = self.a_abs * self.multiplier * dw
        return step
