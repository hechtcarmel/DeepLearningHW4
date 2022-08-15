from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calc_step(self):
        raise NotImplementedError

    @staticmethod
    def get_params(self):
        return self.__init__.__code__.co_varnames[1:self.__init__.__code__.co_argcount]

    @property
    def name(self):
        return self.__name__
