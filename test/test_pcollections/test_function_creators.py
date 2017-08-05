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
    assert fun(2) == 2
    assert fun(2) == 1


def test_InverseDecay():
    fun = InverseDecay(1 * 3 * 5 * 7, 2)
    assert fun(1) == 1 * 3 * 5 * 7
    assert fun(1000) == 1 * 5 * 7
    assert fun(125) == 1 * 3 * 7
    assert fun("toi") == 1 * 3 * 5


def test_all():
    fun = StepFun(ExponentialDecay(2), 2)
    assert fun(0.1) == 0.1
    assert fun(0.1) == 0.1 * np.exp(-2)
