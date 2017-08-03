from collections import defaultdict


class NObject(object):

    def __init__(self, dim_in, dim_out):
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._memory = None
        self._use_memory = True

        self.reset_memory()

    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        return self._dim_out

    def use_memory(self, use=True):
        self._use_memory = use

    def memorize(self, key, item):
        """Store item in memory at key."""
        if self._use_memory:
            self._memory[key].append(item)

    def read_memory(self, key, last=False):
        """Return memory associated to key.

        Args;
            key (str)
            last (boolean, optional): whether you want to access only last item
        """
        if last:
            return self._memory[key][-1]
        else:
            return self._memory[key]

    def reset_memory(self):
        """Empty memory."""
        self._memory = defaultdict(list)

    # To Implement

    def check(self):
        """Check whether object correctly defined."""
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return NotImplementedError

    def update(self, *args, **kwargs):
        return NotImplementedError

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

    def use_memory(self, use=True):
        super().use_memory(use=use)
        for elem in self.iter():
            elem.use_memory(use=use)

    def update(self, *args, **kargs):
        for elem in iter(self):
            elem.update()

    def __iter__(self):
        return iter(self._elements)

    def iter(self):
        return iter(self)

    def riter(self):
        return iter(reversed(self._elements))

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
