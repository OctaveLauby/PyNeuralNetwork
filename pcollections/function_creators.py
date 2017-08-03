import numpy as np

from utils.function import Function


def exponential_decay(k):
    """Return exponential exp(-kt) function."""
    return Function(lambda x: x*np.exp(-k))


class StepDecay(Function):

    def __init__(self, fun, step):
        super().__init__(fun)
        self.step = step
        self.count = 0

    def __call__(self, x):
        self.count += 1
        if self.count >= self.step:
            return super().__call__(x)
        else:
            return x


def step_decay(n, factor):
    """Multiply at every n steps by factor."""
    return StepDecay(lambda x: factor * x, n)


class InverseDecay(Function):

    def __init__(self, alpha, k):
        super().__init__(None)
        self.alpha = alpha
        self.k = k
        self.iteration = -1

    def __call__(self, x):
        self.iteration += 1
        return self.alpha / (1 + self.k * self.iteration)


def inverse_decay(alpha, k):
    """Return alpha / 1 + (k*iteration)"""
    return InverseDecay(alpha, k)
