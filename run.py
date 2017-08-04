"""Run a network training on a dataset.

python run.py -f data\iris.csv -l class -n 0 -r 0.001 -m 0.9
"""

import argparse

from network import HNN
from pcollections.function_creators import (
    exponential_decay,
    step_decay,
    inverse_decay,
)
from utils.dataset import DataSet


def identity(x):
    return x


def main(csv_path, label_col, hidden_layers, learning_kwargs):
    # ----------------------------------------------------------------------- #
    # DataSet reading
    ds = DataSet(csv_path, label_col=label_col)
    input_set = ds.input_set
    output_set = ds.output_set

    print("**** Data :")
    ds.display_struct()
    print()

    # ----------------------------------------------------------------------- #
    # Initialization
    # Network
    hidden_layers_nN = [ds.dim_out for layer_i in range(hidden_layers)]
    network = HNN(
        dim_in=ds.dim_in,
        dim_out=ds.dim_out,
        hidden_layers_nN=hidden_layers_nN,
    )

    print("**** Original Network :")
    network.pprint()
    print()

    # ----------------------------------------------------------------------- #
    # Learning

    # Training
    network.fit(input_set, output_set, **learning_kwargs)

    # Test
    predictions = network.predict(input_set)
    print(
        "Rights : %.1f %%"
        % (100 * ds.rights_ratio(predictions, output_set))
    )
    print()

    print("**** Trained Network :")
    network.pprint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a neural network and train it on dataset."
    )

    # ---- DataSet
    parser.add_argument(
        "-f", "--file_path", type=str,
        required=True,
        help=(
            "path to data csv file."
        ),
    )
    parser.add_argument(
        "-l", "--label_fieldname", type=str,
        required=True,
        help=(
            "column fieldname where to read label."
        ),
    )

    # ---- Neural Network
    parser.add_argument(
        "-n", "--hidden_layers", type=int,
        required=False, default=1,
        help=(
            "number of hidden layers in network,"
            " default is 1"
        ),
    )

    # ---- Learning args
    parser.add_argument(
        "-r", "--learning_rate", type=float,
        required=False, default=0.001,
        help=(
            "initial learning rate,"
            " default is 0.001"
        ),
    )
    parser.add_argument(
        "-m", "--momentum", type=float,
        required=False, default=0.9,
        help=(
            "momentum,"
            " default is 0.9"
        ),
    )
    parser.add_argument(
        "-d", "--decay", choices=['id', 'exp', 'step', 'inv'],
        required=False, default='id',
        help=(
            "decay function, set k for decay rate,"
            " default is 'id'"
        ),
    )
    parser.add_argument(
        "-k", "--decay_rate", type=float,
        required=False, default=0.1,
        help=(
            "decay function rate,"
            " default is 0.1"
        ),
    )
    args = parser.parse_args()
    csv_path = args.file_path  # "data/iris.csv"
    label_col = args.label_fieldname  # "class"
    hidden_layers = args.hidden_layers
    learning_rate = args.learning_rate
    momentum = args.momentum

    k = 0.01
    functions = {
        'id': identity,
        'exp': exponential_decay(k),
        'step': step_decay(50, k),
        'inv': inverse_decay(learning_rate, k),
    }

    learning_kwargs = {
        'learning_rate': learning_rate,
        'momentum': momentum,

        'batch_size': 1,
        'iterations': 300,

        'verbose_lvl': 2,
        'verbose_step': 50,
    }
    main(csv_path, label_col, hidden_layers, learning_kwargs)
