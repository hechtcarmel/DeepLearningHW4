import torch
from optimizers.Optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = (0, 0)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 0

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
