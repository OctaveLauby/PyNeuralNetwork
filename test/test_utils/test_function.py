from utils.function import Function


def compare(f1, f2):
    for e in [-10, -5, -1.5, 0, 1.5, 5, 10]:
        assert f1(e) == f2(e)
    return True


def test_Function():
    f1 = Function(lambda x: x+2)
    f2 = lambda x: x-2

    assert compare(f1(f2), lambda x: x)
    assert compare(f1 * f2, lambda x: (x - 2)*(x + 2))
    assert compare(f1 + f2, lambda x: 2 * x)
    assert compare(f1 - f2, lambda x: 4)
    assert compare(f1 / f2, lambda x: (x + 2) / (x - 2))

    assert compare(f2 * f1, f1 * f2)
    assert compare(f2 + f1, f1 + f2)
    assert compare(f2 - f1, lambda x: - 4)
    assert compare(f2 / f1, lambda x: (x - 2) / (x + 2))
