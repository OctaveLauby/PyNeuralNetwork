import numpy as np

from pcollections.function_creators import (
    ExponentialDecay,
    InverseDecay,
    StepFun,
)


def test_ExponentialDecay():
    fun = ExponentialDecay(2)
    assert fun(0.1) == 0.1 * np.exp(-2)
    assert fun(0) == 0


def test_StepFun():
    fun = StepFun(lambda x: x * 0.5, 2)
    assert fun(2) == 2
    assert fun(2) == 1


def test_InverseDecay():
    fun = InverseDecay(6, 2)
    assert fun(1) == 6
    assert fun(0) == 2
