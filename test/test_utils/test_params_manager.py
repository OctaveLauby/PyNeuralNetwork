import pytest

from utils.params_manager import read_params


def test_read_params():
    dft_params = {
        1: "un",
        2: "deux",
        3: "trois",
    }

    params_1 = {
        1: "one",
        3: None,
    }
    assert read_params(params_1, dft_params) == {
        1: "one",
        2: "deux",
        3: "trois",
    }

    params_2 = {
        2: "dos",
        3: "tres",
    }
    assert read_params(params_2, dft_params) == {
        1: "un",
        2: "dos",
        3: "tres",
    }

    params_3 = {
        0: "zero",
        3: "troyes",
        4: "quatre",
    }
    with pytest.raises(KeyError):
        read_params(params_3, dft_params)
