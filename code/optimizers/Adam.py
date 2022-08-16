import torch
from optimizers.Optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, args):
        self.m_dw, self.v_dw = (0, 0)
        self.beta1 = getattr(args, 'beta1', 0.9)
        self.beta2 = getattr(args, 'beta2', 0.999)
        self.epsilon = getattr(args, 'epsilon', 1e-8)
        self.eta = getattr(args, 'eta', 0.01)
        self.t = 0
        super().__init__()

    def calc_step(self, w, dw=None):
        # dw, db are from current minibatch
        # momentum beta 1
        with torch.no_grad():
            self.t += 1
            dw = w.grad if dw is None else dw
            # * weights * #
            self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw

            # rms beta 2
            # * weights * #
            self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw * 2)
            # bias correction
            m_dw_corr = self.m_dw / (1 - self.beta1 ** self.t)
            v_dw_corr = self.v_dw / (1 - self.beta2 ** self.t)
            step = self.eta * (m_dw_corr / (torch.sqrt(v_dw_corr) + self.epsilon))
        return step
