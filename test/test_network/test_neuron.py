import numpy as np

from network.neuron import Neuron


def test_Neuron():

    act_fun = lambda x: x*x
    act_der = lambda x: 2

    def init_fun(i):
        if i is -1:
            return 0.5
        else:
            return (i+1)

    neuron = Neuron(
        dim_in=3,
        act_fun=act_fun,
        act_der=act_der,
        init_fun=init_fun,
    )
    neuron._test = True

    def output(vector):
        return act_fun(wsum(vector))

    def wsum(vector):
        return vector[0] + 2*vector[1] + 3*vector[2] + 0.5

    vector_1 = np.array([1, 2, 3])
    vector_2 = np.array([1, 0, 3])
    assert neuron.forward(vector_1) == output(vector_1)
    assert neuron.forward(vector_2) == output(vector_2)
    assert neuron.read_memory('input') == [vector_1, vector_2]
    assert neuron.read_memory('wsum') == [wsum(vector_1), wsum(vector_2)]
