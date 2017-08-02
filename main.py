import numpy as np
import random

from pcollections.functions import euclidean_dist, euclidean_dist_jac
from network import NNetwork, HNLayer


cost_fun = euclidean_dist
cost_jac = euclidean_dist_jac


def act_fun(value):
    return value


def act_der(value):
    return 1


def init_fun(i):
    return random.random()


network = NNetwork(dim_in=3, dim_out=2, cost_fun=cost_fun, cost_jac=cost_jac)

n_kwargs = {
    'dim_in': 3,
    'act_fun': act_fun,
    'act_der': act_der,
    'init_fun': init_fun,
}
layer_1 = HNLayer(dim_in=3, nN=3, n_kwargs=n_kwargs)
layer_2 = HNLayer(dim_in=3, nN=2, n_kwargs=n_kwargs)

network.add(layer_1)
network.add(layer_2)

network.check()
network.pprint()

print(network.forward(np.array([1, 1, 1])))
