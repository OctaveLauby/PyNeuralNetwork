import numpy as np

from .base import NObject


class ShortCutted(Exception):
    pass


class Neuron(NObject):
    """Neuron."""

    def __init__(self, dim_in, act_fun, act_der, init_fun):
        """Create a neuron.

        Args:
            dim_in (int): dim of input for neuron
            act_fun (callable): activation function
                float -> float
            act_der (callable): derivative of activation function
                float -> float
            init_fun (callable): function to create weights and bias
                int (index of weight, -1 for bias) -> float
        """
        super().__init__(dim_in, 1)
        self._act_fun = act_fun
        self._act_der = act_der
        self._init_fun = init_fun

        self._weights = None
        self._bias = None
        self._init_weights()

    def _init_weights(self):
        """Initialise weights and bias using init_fun."""
        self._bias = self._init_fun(-1)
        self._weights = np.array([
            self._init_fun(i) for i in range(self.dim_in)
        ])

    @property
    def bias(self):
        return self._bias

    @property
    def weights(self):
        return self._weights

    def check(self):
        return True

    def backward(self):
        # return self.weights * self._act_der(self._last_input)
        raise NotImplementedError

    def forward(self, vector):
        return self.compute(vector)

    def update(self, *args, **kargs):
        raise NotImplementedError

    # Calculation

    def compute(self, vector):
        """Return """
        wsum = self.weighted_sum(vector)
        self._memory['wsum'].append(wsum)
        return self._act_fun(wsum)

    def delta(self, nl_i_weights, nl_delta, weighted_sum):
        """Error due to neuron.

        Args:
            nl_i_weights (np.array, size=nN_nl):
                weights of next layer regarding this neuron output
            nl_delta (np.array, size=nN_nl):
                deltas of next layer
            weighted_sum (float):
                weighted_sum that was calculated

        Returns:
            (float)
        """
        raise ShortCutted
        return (
            np.dot(nl_i_weights, nl_delta) * self._act_der(weighted_sum)
        )

    def rate_of_change_bias(self, delta):
        """Rate of change with respect of bias."""
        raise ShortCutted
        return delta

    def rate_of_change_weights(self, vector, delta_w):
        """Rate of change with respect of weights."""
        return vector * delta_w

    def weighted_sum(self, vector):
        """Weighted sum of vector comp."""
        return np.dot(self.weights, vector) + self._bias

    def __repr__(self):
        return (
            "<Neuron {}+{:.2e}>"
            .format(["%.2e" % weig for weig in self.weights], self.bias)
        )
