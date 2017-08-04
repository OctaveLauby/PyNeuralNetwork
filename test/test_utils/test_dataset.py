from utils.dataset import BaseDataSet, DataSet


def test_BaseDataSet():
    size = 3 * 10
    input_set = [
        [i, i, i, i] for i in range(1, size+1)
    ]
    input_labels = [
        str(i % 3) for i in range(1, size + 1)
    ]
    ds = BaseDataSet(input_set, input_labels, dim_in=4)
    ds.display_data()
    assert ds.size == size
    assert ds.dim_in == 4
    assert ds.dim_out == 3
    assert ds.labels == [str(i) for i in range(3)]
    assert [list(map(float, output)) for output in ds.output_set] == [
        [
            1. if int(label) == 0 else 0.,
            1. if int(label) == 1 else 0.,
            1. if int(label) == 2 else 0.,
        ]
        for label in input_labels
    ]

    ds1, ds2 = ds.split(0.8)
    assert ds1.size == 8 * size // 10
    assert ds2.size == 2 * size // 10
