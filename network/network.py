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
                output (np.array), expected_output (np.array) -> cost (float)
            cost_jac (callable): derivate of cost_fun (shared by each
                partial derivates)
        """
        super().__init__(dim_in=dim_in, dim_out=dim_out, elem_cls=NLayer)
        self._cost_fun = cost_fun
        self._cost_jac = cost_jac

    def check(self):
        """Make sure dimension chain is consistent."""
        expected_dim = self.dim_in
        for layer in self.iter():
            assert layer.dim_in == expected_dim
            layer.check()
            expected_dim = layer.dim_out
        assert self.dim_out == expected_dim
        return True

    def cost(self, expected_output):
        output = self.read_memory('output', last=True)
        return self._cost_fun(output, expected_output)

    def backward(self, expected_output):
        """STEP-2: Back-propagate error on last output."""
        output = self.read_memory('output', last=True)
        nl_delta = self._cost_jac(output, expected_output)
        nl_weights = 1
        for layer in self.riter():
            nl_delta = layer.backward(
                nl_weights=nl_weights,
                nl_delta=nl_delta,
            )
            nl_weights = layer.weights()

    def forward(self, vector):
        """STEP-1: Forward input vector to calculate output."""
        self.memorize('input', vector)
        int_vect = vector
        for layer in self.iter():
            int_vect = layer.forward(vector=int_vect)
        output = int_vect
        self.memorize('output', output)
        return output

    def update(self, learning_rate, momentum=0):
        """STEP-3: Update weights using last batch of inputs.

        Args:
            learning_rate (float)
            momentum (float): usually around 0.9
        """
        for layer in self.iter():
            layer.update(learning_rate, momentum)
        self.reset_memory()

    @property
    def nL(self):
        return self.nE
