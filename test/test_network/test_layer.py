import numpy as np

from network.layer import HNLayer


def test_HNLayer():

    bias = 0.5

    def act_fun(value):
        return 2 * value

    def act_der(value):
        return 2

    def init_fun(i):
        if i is -1:
            return bias
        return i

    n_kwargs = {
        'dim_in': 2,
        'act_fun': act_fun,
        'act_der': act_der,
        'init_fun': init_fun,
    }
    layer_1 = HNLayer(dim_in=2, nN=2, n_kwargs=n_kwargs)

    x0 = np.array([1, 1])
    v1y = act_fun(np.dot(x0, np.array([0, 1])) + bias)
    assert list(layer_1.forward(x0)) == [v1y, v1y]
