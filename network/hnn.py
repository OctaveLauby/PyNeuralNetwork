
import random

from pcollections.functions import (
    euclidean_dist, euclidean_dist_jac,
    sigmoid, sigmoid_der,
    tanh, tanh_der
)

from .layer import HNLayer
from .network import NNetwork


class HNN(NNetwork):
    """Fully Connected Homogeneous Neural Network."""

    def __init__(self, dim_in, dim_out, cost_fun=None, cost_jac=None,
                 nHL=None, hidden_layers_nN=[], act_fun=None, act_der=None,
                 outact_fun=None, outact_der=None,
                 init_fun=None,
                 ):
        """Create a network.


        Args:
            dim_in (int): dimension of input space
            dim_out (int): dimension of output space
            cost_fun (callable): cost function
                output (np.array), expected_output (np.array) -> cost (float)
                > default is euclidean distance
            cost_jac (callable): jacobian of cost_fun

            nHL (int): number of hidden layers
            hidden_layers_nN (list of int): number of neurons per
                hidden layers
                /!\\ if nHL is given, nHL prevails
                > default is no hidden layer
            act_fun (callable): activation fun of hidden layers
                > default is sigmoid
            act_der (callable): act_fun der of hidden layers

            outact_fun (callable): activation fun of output layer
                > default is tanh
            outact_der (callable): act_fun der of output layer

            init_fun (callable): initialisation fun of weights
                | int (neuron_i) -> float
                > default is weight = random_float(-1, 1) & bias = 0
        """
        # Init network
        if cost_fun is None:
            cost_fun = 0.5 * euclidean_dist
            cost_jac = 0.5 * euclidean_dist_jac
        elif cost_jac is None:
            raise ValueError("cost_jac is given and cost_fun is not.")
        super().__init__(dim_in, dim_out, cost_fun=cost_fun, cost_jac=cost_jac)

        # Create hidden layers
        if act_fun is None:
            act_fun = sigmoid
            act_der = sigmoid_der
        elif act_der is None:
            raise ValueError("act_der is given when act_fun is not.")
        if init_fun is None:
            def init_fun(i):
                """Random float between -1 and 1"""
                if i == -1:
                    return 0.
                return 2 * random.random() - 1
        last_dim = dim_in
        n_kwargs = {
            'act_fun': act_fun,
            'act_der': act_der,
            'init_fun': init_fun,
        }

        if nHL:
            hidden_layers_nN = self.smart_hlayers_nN(nHL)
        for layer_nN in hidden_layers_nN:
            self.add(HNLayer(dim_in=last_dim, nN=layer_nN, **n_kwargs))
            last_dim = layer_nN

        # Create last layer
        if outact_fun is None:
            outact_fun = tanh
            outact_der = tanh_der
        elif outact_der is None:
            raise ValueError("outact_der is given when act_fun is not.")
        n_kwargs = {
            'act_fun': outact_fun,
            'act_der': outact_der,
            'init_fun': init_fun,
        }
        self.add(HNLayer(dim_in=last_dim, nN=dim_out, **n_kwargs))

        self.check()

    def smart_hlayers_nN(self, nHL):
        """Return a smart configuration on neurons per layers."""
        hidden_layers_nN = []
        delta_dim = self.dim_out - self.dim_in
        for layer_i in range(nHL):
            nN = self.dim_in + round(delta_dim * (layer_i + 1) / (nHL + 1))
            hidden_layers_nN.append(int(nN))
        return hidden_layers_nN
