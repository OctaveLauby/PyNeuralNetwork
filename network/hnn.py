
import random

from collection import ACT_FUN_DER, COST_FUN_DER
from utils.params_manager import read_params

from .layer import HNLayer
from .network import NNetwork


def default_init_fun(i):
    """Random float between -1 and 1"""
    if i == -1:
        return 0.01  # in case of relu activation func, to avoid dead neurons
    return 2 * random.random() - 1


class HNN(NNetwork):
    """Fully Connected Homogeneous Neural Network."""

    dft_build_params = {
        'cost_fun': 'euclidean',
        'nHL': 1,
        'hlayers_nN': [],
        'act_fun': "sigmoid",
        'outact_fun': "tanh",
        'init_fun': default_init_fun,
    }

    def __init__(self, dim_in, dim_out, build_params):
        """Create a network.

        Args:
            dim_in (int): dimension of input space
            dim_out (int): dimension of output space
            cost_fun (str): cost function name
            build_params (dict): parameters to build network
                @see HNN.dft_build_params for defaults
                cost_fun    (str)       cost function name
                nHL         (int)       number of hidden layers
                hlayers_nN  (int-list)  number of neurons per hidden layer
                    /!\\ if nHL is given, nHL prevails
                act_fun     (str)       activation fun name of hidden layers
                outact_fun  (str)       activation fun name of output layer
                init_fun    (callable)  initialisation fun of weights
        """
        # Init network
        self._build_params = read_params(build_params, HNN.dft_build_params)

        cost_fun, cost_jac = COST_FUN_DER[self._build_params['cost_fun']]
        super().__init__(dim_in, dim_out, cost_fun=cost_fun, cost_jac=cost_jac)

        self.build()

        self.check()

    def build(self):
        """Build layers."""
        # Create hidden layers
        act_fun, act_der = ACT_FUN_DER[self.build_param('act_fun')]
        n_kwargs = {
            'act_fun': act_fun,
            'act_der': act_der,
            'init_fun': self.build_param('init_fun'),
        }

        last_dim = self.dim_in
        hlayers_nN = self.build_param('hlayers_nN')
        if self.build_param('nHL'):
            hlayers_nN = self.smart_hlayers_nN(self.build_param('nHL'))
        for layer_nN in hlayers_nN:
            self.add(HNLayer(dim_in=last_dim, nN=layer_nN, **n_kwargs))
            last_dim = layer_nN

        # Create output layer
        outact_fun, outact_der = ACT_FUN_DER[self.build_param('outact_fun')]
        n_kwargs = {
            'act_fun': outact_fun,
            'act_der': outact_der,
            'init_fun': self.build_param('init_fun'),
        }
        self.add(HNLayer(dim_in=last_dim, nN=self.dim_out, **n_kwargs))

        self.check()

    def build_param(self, key):
        """Return building parameter."""
        return self._build_params[key]

    # -----------------------------------------------------------------------#
    # Utils

    def smart_hlayers_nN(self, nHL):
        """Return a smart configuration on neurons per layers."""
        hlayers_nN = []
        delta_dim = self.dim_out - self.dim_in
        for layer_i in range(nHL):
            nN = self.dim_in + round(delta_dim * (layer_i + 1) / (nHL + 1))
            hlayers_nN.append(int(nN))
        return hlayers_nN

    def str_params(self):
        res = ""
        max_length = max([len(key) for key in self._build_params])
        for key, value in self._build_params.items():
            res += (
                "\n\t {key}\t{value}".format(
                    key=key + " " * (max_length - len(key)),
                    value=value
                )
            )
        return res[1:]  # skip first new line

    def display_params(self):
        """Display params used to build network."""
        print(self.str_params())
