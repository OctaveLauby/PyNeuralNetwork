import numpy as np

from pcollections.functions import euclidean_dist


def test_euclidean_dist():
    v1 = np.array([4, 5])
    v2 = np.array([1, 1])
    assert euclidean_dist(v1, v2) == 5
