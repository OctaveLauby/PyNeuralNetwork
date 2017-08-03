import numpy as np

from pcollections.function_creators import (
    exponential_decay,
    step_decay,
    inverse_decay,
)


def test_exponential_decay():
    fun = exponential_decay(2)
    assert fun(1) == np.exp(-2)
    assert fun(0) == 0


def test_step_decay():
    fun = step_decay(2, 0.5)
    assert fun(2) == 2
    assert fun(2) == 1


def test_inverse_decay():
    fun = inverse_decay(6, 2)
    assert fun(1) == 6
    assert fun(0) == 2
