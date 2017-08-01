import numpy as np

from .base import NObject


class Neuron(NObject):
    """Neuron."""

    def __init__(self, dim_in, act_fun, act_der, init_fun):
        super().__init__(dim_in, 1)
        self._act_fun = act_fun
        self._act_der = act_der
        self._init_fun = init_fun

        self._weights = None
        self._init_weights()

    def _init_weights(self):
        self._weights = np.array([
            self._init_fun() for i in range(self.dim_in)
        ])

    def check(self):
        return True

    def __repr(self):
        return "<Neuron{}".format(list(self._weights))
