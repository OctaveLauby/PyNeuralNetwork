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

        self._learning_speed = np.array([0] * dim_in)
        self._learning_speed_b = 0

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

    def backward(self, nl_i_weights, nl_delta):
        raise ShortCutted
        delta = self.delta(nl_i_weights, nl_delta)
        self.memorize('delta', delta)
        return delta

    def forward(self, vector):
        return self.compute(vector)

    def update(self, learning_rate, momentum):
        """Update weights, bias and reset memory."""
        # Capture last batch
        inputs = self.read_memory('input')
        deltas = self.read_memory('delta')
        weights_rates = [
            self.rate_of_change_weights(vector, delta)
            for vector, delta in zip(inputs, deltas)
        ]
        bias_rates = [
            self.rate_of_change_bias(delta)
            for delta in deltas
        ]

        # Compute a rate given former rates (mean)
        weights_rate = sum(weights_rates) / len(weights_rates)
        bias_rate = sum(bias_rates) / len(bias_rates)

        # Compute learning rates
        self._learning_speed = (
            momentum * self._learning_speed
            - learning_rate * weights_rate
        )
        self._learning_speed_b = (
            momentum * self._learning_speed
            - learning_rate * bias_rate
        )

        # Update
        self._weights = self._weights + self._learning_speed
        self._bias = self._bias + self._learning_speed_b
        self.reset_memory()

    # Calculation

    def compute(self, vector):
        """Return """
        wsum = self.weighted_sum(vector)
        self.memorize('wsum', wsum)
        return self._act_fun(wsum)

    def delta(self, nl_i_weights, nl_delta, weighted_sum=None):
        """Error due to neuron.

        Args:
            nl_i_weights (np.array, size=nN_nl):
                weights of next layer regarding this neuron output
            nl_delta (np.array, size=nN_nl):
                deltas of next layer
            weighted_sum (float, optional):
                weighted_sum that was calculated

        Returns:
            (float)
        """
        raise ShortCutted
        if weighted_sum is None:
            weighted_sum = self.read_memory('wsum', last=True)
        return (
            np.dot(nl_i_weights, nl_delta) * self._act_der(weighted_sum)
        )

    def rate_of_change_bias(self, delta):
        """Rate of change with respect of bias."""
        return delta

    def rate_of_change_weights(self, vector, delta):
        """Rate of change with respect of weights."""
        return vector * delta

    def weighted_sum(self, vector):
        """Weighted sum of vector comp."""
        return np.dot(self.weights, vector) + self._bias

    def __repr__(self):
        return (
            "<Neuron {}+{:.2e}>"
            .format(["%.2e" % weig for weig in self.weights], self.bias)
        )
