class Function(object):

    def __init__(self, fun):
        self.fun = fun

    def __add__(self, other):
        if callable(other):
            def fun(*args, **kwargs):
                return self(*args, **kwargs) + other(*args, **kwargs)
        else:
            def fun(*args, **kwargs):
                return self.fun(*args, **kwargs) + other
        return Function(fun)
        return Function(fun)

    def __mul__(self, other):
        if callable(other):
            def fun(*args, **kwargs):
                return self(*args, **kwargs) * other(*args, **kwargs)
        else:
            def fun(*args, **kwargs):
                return self.fun(*args, **kwargs) * other
        return Function(fun)

    def __neg__(self):
        def fun(*args, **kwargs):
            return - self(*args, **kwargs)
        return Function(fun)

    def __sub__(self, other):
        if callable(other):
            def fun(*args, **kwargs):
                return self(*args, **kwargs) - other(*args, **kwargs)
        else:
            def fun(*args, **kwargs):
                return self.fun(*args, **kwargs) - other
        return Function(fun)

    def __truediv__(self, other):
        if callable(other):
            def fun(*args, **kwargs):
                return self(*args, **kwargs) / other(*args, **kwargs)
        else:
            def fun(*args, **kwargs):
                return self.fun(*args, **kwargs) / other
        return Function(fun)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (- self) + other

    def __rtruediv__(self, other):
        return Function.__truediv__(other, self)

    def __call__(self, *args, **kwargs):
        if len(args) is 1 and not kwargs:
            arg = args[0]
            if callable(arg):
                def fun(*args, **kwargs):
                    return self(arg(*args, **kwargs))
                return Function(fun)
        return self.fun(*args, **kwargs)
