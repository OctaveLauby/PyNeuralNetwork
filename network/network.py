from .base import NContainer
from .layer import NLayer


class NNetwork(NContainer):
    """Neural Network using backpropagation algorithm."""

    def __init__(self, dim_in, dim_out, cost_fun):
        super().__init__(dim_in=dim_in, dim_out=dim_out, elem_cls=NLayer)
        self._cost_fun = cost_fun

    def check(self):
        """Make sure dimension chain is consistent."""
        excepted_dim = self.dim_in
        for layer in iter(self):
            assert layer.dim_in == excepted_dim
            layer.check()
            excepted_dim = layer.dim_out
        assert self.dim_out == excepted_dim
        return True

    def _forward(self, vector):
        int_vect = vector
        for elem in iter(self):
            int_vect = elem.forward(int_vect)
        output = int_vect
        return output

    @property
    def nL(self):
        return self.nE
