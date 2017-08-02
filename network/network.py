from .base import NContainer
from .layer import NLayer


class NNetwork(NContainer):
    """Neural Network using backpropagation algorithm."""

    def __init__(self, dim_in, dim_out, cost_fun, cost_jac):
        """Create a neural network.

        Args:
            dim_in (int): dimension of input space
            dim_out (int): dimension of output space
            cost_fun (callable): function used to calculate cost
            cost_der (callable): derivate of cost_fun (shared by each
                partial derivates)
        """
        super().__init__(dim_in=dim_in, dim_out=dim_out, elem_cls=NLayer)
        self._cost_fun = cost_fun
        self._cost_jac = cost_jac

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
