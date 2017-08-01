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

    def check(self):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def foward(self, *args, **kwargs):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    # Pretty display
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
        print(self.pstring())

    def pstring(self):
        return repr(self)


class NContainer(NObject):

    def __init__(self, dim_in, dim_out, elem_cls):
        super().__init__(dim_in, dim_out)
        self._NE = 0
        self._elements = []
        self._elem_cls = elem_cls

    def add(self, elem):
        self._elements.append(elem)

    def insert(self, index, elem):
        self._elements.insert(index, elem)

    @property
    def indexes(self):
        return range(self.nE)

    @property
    def nE(self):
        return len(self._elements)

    def backward(self, *args, **kwargs):
        output = []
        for elem in iter(self):
            output.append(elem.backard(*args, **kwargs))
        return np.array(output)

    def forward(self, vector):
        assert len(vector) == self.dim_in
        int_vect = vector
        for elem in iter(self):
            int_vect = elem.forward(int_vect)
        output = int_vect
        assert len(output) == self.dim_out
        return output

    def update(self):
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
        return (
            str(self)
            + (
                ("\n" if self.nE else "")
                + "\n".join(map(lambda elem: elem.pstring(), iter(self)))
            ).replace("\n", "\n\t")
        )
