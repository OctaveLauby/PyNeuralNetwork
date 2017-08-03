import numpy as np
import random

from pcollections.functions import (
    euclidean_dist, euclidean_dist_jac,
    sigmoid, sigmoid_der,
)
from network import NNetwork, HNLayer

# --------------------------------------------------------------------------- #
# Initialization

# Network
cost_fun = euclidean_dist
cost_jac = euclidean_dist_jac
network = NNetwork(dim_in=3, dim_out=2, cost_fun=cost_fun, cost_jac=cost_jac)

# Neurons
n_kwargs = {
    'act_fun': sigmoid,
    'act_der': sigmoid_der,
    'init_fun': lambda i: random.random(),
}

# Layers
layer_1 = HNLayer(dim_in=3, nN=3, **n_kwargs)
layer_2 = HNLayer(dim_in=3, nN=2, **n_kwargs)
network.add(layer_1)
network.add(layer_2)

network.check()
print("**** Original Network :")
network.pprint()


# --------------------------------------------------------------------------- #
# Learning
vector = np.array([1, 1, 1])
expected_output = np.array([0, 1])
learning_rate = 0.1
momentum = 0.9
repeat = 1000

# Forwarding
print("\n**** Forwarding :", vector)
print("> output :", network.forward(vector))

# Back Propagation
print("\n**** Back propagation with %s." % expected_output)
print("> cost :", network.cost(expected_output))
network.backward(expected_output)
network.update(learning_rate=learning_rate, momentum=momentum)

print("\n**** Updated network :")
network.pprint()

print("\n**** Repeat it %s times :" % repeat)
for i in range(repeat):
    res = network.forward(vector)
    network.backward(expected_output)
    print(res, network.cost(expected_output))
    network.update(learning_rate=learning_rate, momentum=momentum)
network.pprint()
