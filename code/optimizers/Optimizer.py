from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self):
        pass

    # @abstractmethod
    # def __call__(self, pert, w, dw=None):
    #     raise NotImplementedError

    @abstractmethod
    def calc_step(self, w, dw=None):
        raise NotImplementedError

    @classmethod
    def get_params(cls):
        params = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
        defaults = cls.__init__.__defaults__
        if defaults is None:
            defaults = [None] * len(params)
        default_params_values = dict(zip(params[-len(defaults):], defaults)) if defaults else {}
        return default_params_values

    @property
    def name(self):
        return self.__name__
