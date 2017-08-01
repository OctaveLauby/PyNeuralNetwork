import numpy as np

from .base import NContainer
from .neuron import Neuron


class NLayer(NContainer):
    """Neural Layer."""

    def __init__(self, dim_in):
        super().__init__(dim_in=dim_in, dim_out=0, elem_cls=Neuron)

    @property
    def nN(self):
        return self.nE

    @property
    def dim_out(self):
        return self.nN

    def check(self):
        """Make sure dimensions are consistent."""
        for neuron in iter(self):
            assert neuron.dim_in == self.dim_in
            assert neuron.check()
        assert self.nN == self.dim_out

    def _forward(self, vector):
        res = []
        for elem in iter(self):
            res.append(elem.forward(vector))
        return np.array(res)


class HNLayer(NLayer):
    """Homogeneous Neural Layer."""

    def __init__(self, dim_in, nN, n_kwargs):
        super().__init__(dim_in=dim_in)
        self._n_kwargs = n_kwargs
        self._init_neurons(nN)

    def _init_neurons(self, number):
        """Create neurons."""
        for i in range(number):
            self.add(Neuron(**self._n_kwargs))
