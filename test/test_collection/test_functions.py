import numpy as np

from collection.functions import (
    euclidean_dist,
    sigmoid,
    sigmoid_der,
)


def test_euclidean_dist():
    v1 = np.array([4, 5])
    v2 = np.array([1, 1])
    assert euclidean_dist(v1, v2) == 5
    assert (0.5 * euclidean_dist)(v1, v2) == 2.5


def test_sigmoid():
    inf = float('inf')
    v = np.array([0, inf, -inf])
    assert (sigmoid(v) == np.array([1/2, 1, 0])).all()
    assert (sigmoid_der(v) == np.array([1/4, 0, 0])).all()
