import numpy as np

from network.layer import HNLayer
from network.network import NNetwork
from pcollections.functions import euclidean_dist, euclidean_dist_jac


def test_NNetwork():

    bias = 0.5

    cost_fun = euclidean_dist
    cost_jac = euclidean_dist_jac

    def act_fun(value):
        return 2 * value

    def act_der(value):
        return 2

    def init_fun(i):
        if i is -1:
            return bias
        return i

    network = NNetwork(
        dim_in=2, dim_out=1, cost_fun=cost_fun, cost_jac=cost_jac
    )

    n_kwargs = {
        'act_fun': act_fun,
        'act_der': act_der,
        'init_fun': init_fun,
    }
    layer_1 = HNLayer(dim_in=2, nN=2, **n_kwargs)
    layer_2 = HNLayer(dim_in=2, nN=1, **n_kwargs)

    network.add(layer_1)
    network.add(layer_2)

    network.check()

    x0 = np.array([1, 1])
    w_vec = np.array([0, 1])
    v1y = act_fun(np.dot(x0, w_vec) + bias)
    x1 = np.array([v1y, v1y])
    assert network.forward(x0)[0] == act_fun(np.dot(x1, w_vec) + bias)
