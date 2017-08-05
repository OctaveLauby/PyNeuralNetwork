import numpy as np

from utils.function import Function


class ExponentialDecay(Function):
    def __init__(self, k):
        super().__init__(lambda x: x*np.exp(-k))


class InverseDecay(Function):

    def __init__(self, alpha, k):
        super().__init__(None)
        self.alpha = alpha
        self.k = k
        self.iteration = -1

    def __call__(self, x):
        self.iteration += 1
        return self.alpha / (1 + self.k * self.iteration)


class Linear(Function):
    def __init__(self, k):
        super().__init__(lambda x: x*k)


class StepFun(Function):
    """Apply call function every n calls, default_fun the rest of the time."""

    def __init__(self, fun, step, default_fun=None):
        super().__init__(fun)
        self.step = step
        self.count = 0
        self._default_fun = default_fun if default_fun else (
            lambda *args, **kwargs: args[0]
        )

    def __call__(self, *args, **kwargs):
        if len(args) is 1 and not kwargs:
            arg = args[0]
            if callable(arg):
                def fun(*args, **kwargs):
                    return self(arg(*args, **kwargs))
                return Function(fun)
        self.count += 1
        if self.count >= self.step:
            self.count = 0
            return super().__call__(*args, **kwargs)
        else:
            return self._default_fun(*args, **kwargs)
