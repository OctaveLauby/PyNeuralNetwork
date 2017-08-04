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

    def backward(self, nl_delta, nl_weights):
        """Return back-propagated delta?

        Args:
            nl_weights (2d-np.array[nN_nl*nW_nl]):  next layer weights
            nl_delta (np.array[nN_nl]):             next layer deltas

        Return:
            (np.array[nN_l])
        """
        delta = self.delta(nl_delta, nl_weights)
        self.memorize('delta', delta)
        for neuron, delta_n in zip(self.iter(), delta):
            # SHORTCUT : neuron backward, so we memorize from here
            neuron.memorize('delta', delta_n)
        return delta

    def forward(self, vector):
        """Forward input vector to each neuron to calculate layer output."""
        res = []
        wsum_all = []
        for neuron in iter(self):
            res.append(neuron.forward(vector))
            wsum_all.append(
                neuron.read_memory('wsum', last=True)
            )
        self.memorize('input', vector)
        self.memorize('wsum', np.array(wsum_all))
        return np.array(res)

    def update(self, learning_rate, momentum):
        for neuron in self.iter():
            neuron.update(learning_rate, momentum)
        self.reset_memory()

    def delta(self, *args, **kwargs):
        raise NotImplementedError


class HNLayer(NLayer):
    """Homogeneous Neural Layer."""

    def __init__(self, dim_in, nN, act_fun, act_der, init_fun):
        super().__init__(dim_in=dim_in)
        self._act_fun = act_fun
        self._act_der = act_der
        self._init_fun = init_fun
        self._n_kwargs = {
            'dim_in': dim_in,
            'act_fun': act_fun,
            'act_der': act_der,
            'init_fun': init_fun,
        }
        self._init_neurons(nN)

    def _init_neurons(self, number):
        """Create neurons."""
        for i in range(number):
            self.add(Neuron(**self._n_kwargs))

    def weights(self):
        res = []
        for neuron in iter(self):
            res.append(neuron.weights)
        return np.array(res)

    # Calculation

    def delta(self, nl_weights, nl_delta, weighted_sum_vect=None):
        """Error due to layer.

        Reminder:
            Number of weights per neuron of next ladder is equal to number of
            neurons of this layer:
                nW_nl = nN_l

        Args:
            nl_weights (2d-np.array[nN_nl*nW_nl]):  next layer weights
            nl_delta (np.array[nN_nl]):             next layer deltas
            weighted_sum_vect (np.array[nW_l]):     weighted sums (per neuron)
                > default is last calculated

        Returns:
            (np.array[nN_l])
        """
        # SHORTCUT : neuron delta
        if weighted_sum_vect is None:
            weighted_sum_vect = self.read_memory('wsum', last=True)
        return (
            np.dot(nl_weights.T, nl_delta)
            * self._act_der(weighted_sum_vect)
        )

    def rate_of_change_bias(self, delta):
        """Bias rate of change.

        Returns:
            (np.array[nN_l])
        """
        return delta

    def rate_of_change_weights(self, vector, delta):
        """Bias rate of weights.

        Returns:
            (2d-np_array[nN_l*nW_l])
        """
        return np.array([
            neuron.rate_of_change_weights(vector, delta_w)
            for neuron, delta_w in zip(iter(self), delta)
        ])
