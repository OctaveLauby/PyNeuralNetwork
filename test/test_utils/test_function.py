from utils.function import Function


def test_Function():
    f1 = Function(lambda x: x+2)
    f2 = Function(lambda x: x-2)
    my_fun = f1 * f2
    assert my_fun(2) == 0
    assert my_fun(-2) == 0
    assert my_fun(0) == -4
    assert (f1 * (1/2))(2) == 2
    assert ((1/2) * f1)(4) == 3
