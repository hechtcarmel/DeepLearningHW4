from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calc_step(self):
        raise NotImplementedError
