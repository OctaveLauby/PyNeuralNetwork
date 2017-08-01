import numpy as np


class NObject(object):

    def __init__(self, dim_in, dim_out):
        self._dim_in = dim_in
        self._dim_out = dim_out

    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        return self._dim_out

    # To Implement

    def check(self):
        """Check whether object correctly defined."""
        raise NotImplementedError

    def _backward(self, *args, **kwargs):
        raise NotImplementedError

    def _forward(self, vector):
        return NotImplementedError

    def _update(self, *args, **kwargs):
        return NotImplementedError

    # Utils

    def backward(self, *args, **kwargs):
        return self._backward(*args, **kwargs)

    def forward(self, vector):
        """Forward vector to build output."""
        return self._forward(vector)

    def update(self, *args, **kwargs):
        return self._update(*args, **kwargs)

    # Display

    def __repr__(self):
        return (
            "<{cls} | dim {dim_in} to {dim_out}>"
            .format(
                cls=self.__class__.__name__,
                dim_in=self.dim_in,
                dim_out=self.dim_out,
            )
        )

    def __str__(self):
        return repr(self)

    def pprint(self):
        """Pretty print."""
        print(self.pstring())

    def pstring(self):
        """Create pretty string."""
        return repr(self)


class NContainer(NObject):

    def __init__(self, dim_in, dim_out, elem_cls):
        super().__init__(dim_in, dim_out)
        self._NE = 0
        self._elements = []
        self._elem_cls = elem_cls

    def add(self, elem):
        """Add elem to elements."""
        self._elements.append(elem)

    def insert(self, index, elem):
        """Insert elem in elements."""
        self._elements.insert(index, elem)

    @property
    def indexes(self):
        return range(self.nE)

    @property
    def nE(self):
        return len(self._elements)

    def _backward(self, *args, **kwargs):
        output = []
        for elem in iter(self):
            output.append(elem.backard(*args, **kwargs))
        return np.array(output)

    def _update(self, *args, **kargs):
        for elem in iter(self):
            elem.update()

    def __iter__(self):
        return iter(self._elements)

    # Pretty display
    def __repr__(self):
        return super().__repr__().replace(
            ">",
            " | {elem_n} {elem_cls}>".format(
                elem_n=self.nE,
                elem_cls=self._elem_cls.__name__,
            )
        )

    def pstring(self):
        """Pretty indented string."""
        return (
            str(self)
            + (
                ("\n" if self.nE else "")
                + "\n".join(map(lambda elem: elem.pstring(), iter(self)))
            ).replace("\n", "\n\t")
        )
