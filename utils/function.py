class Function(object):

    def __init__(self, fun):
        self.fun = fun

    def __mul__(self, other):
        if isinstance(other, Function):
            def fun(*args, **kwargs):
                return other(*args, **kwargs) * self.fun(*args, **kwargs)
            return fun
        else:
            def fun(*args, **kwargs):
                return other * self.fun(*args, **kwargs)
            return fun

    def __rmul__(self, other):
        return self * other

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)
