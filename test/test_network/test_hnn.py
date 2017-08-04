from network.hnn import HNN


def test_HNN():

    network = HNN(
        dim_in=8,
        dim_out=2,
        nHL=3,
    )

    assert network.layers_nN() == [6, 5, 4, 2]
