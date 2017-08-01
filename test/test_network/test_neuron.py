import numpy as np

from network.neuron import Neuron


def test_Neuron():
    dim_in = 3

    def act_fun(x):
        return 2*x

    def act_der(x):
        return 2

    def init_fun(i):
        if i is -1:
            return 0.5
        else:
            return (i+1) / 10

    neuron = Neuron(
        dim_in=dim_in,
        act_fun=act_fun,
        act_der=act_der,
        init_fun=init_fun,
    )

    assert neuron.forward(np.array([10, 20, 30])) == 29.
