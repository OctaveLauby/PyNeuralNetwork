"""Run a network training on a dataset.

python run.py -f data\iris.csv -l class -n 0 -i 300 -r 0.001 -m 0.9
python run.py -f data\iris.csv -l class -n 2 -i 300 -r 0.01 -m 0.9
"""

import argparse

from network import HNN
from pcollections.function_creators import (
    ExponentialDecay,
    InverseDecay,
    StepFun,
)
from utils.dataset import DataSet


def identity(x):
    return x


def main(csv_path, label_col, hidden_layers, learning_kwargs):
    # ----------------------------------------------------------------------- #
    # DataSet reading
    ds = DataSet(csv_path, label_col=label_col)
    training_set, gene_set = ds.split(0.8)

    print("**** Data :")
    ds.display_struct()
    print()

    print("**** Training Set :")
    training_set.display_struct()
    print()

    print("**** Generalisation Set :")
    gene_set.display_struct()
    print()

    # ----------------------------------------------------------------------- #
    # Initialization
    # Network
    network = HNN(
        dim_in=ds.dim_in,
        dim_out=ds.dim_out,
        nHL=hidden_layers,
    )

    print("**** Original Network :")
    network.pprint()
    print()

    # ----------------------------------------------------------------------- #
    # Learning

    # Training
    network.fit(
        training_set.input_set,
        training_set.output_set,
        **learning_kwargs
    )

    # Test
    predictions = network.predict(training_set.input_set)
    print(
        "Rights on training set: %.1f %%"
        % (100 * ds.rights_ratio(predictions, training_set.output_set))
    )
    predictions = network.predict(gene_set.input_set)
    print(
        "Rights on gene set: %.1f %%"
        % (100 * ds.rights_ratio(predictions, gene_set.output_set))
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
        "-i", "--iterations", type=int,
        required=False, default=100,
        help=(
            "number of iterations,"
            " default is 100"
        ),
    )
    parser.add_argument(
        "-b", "--batch_size", type=int,
        required=False, default=1,
        help=(
            "size of batch,"
            " default is 1"
        ),
    )
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

    # ---- Verbose
    parser.add_argument(
        "-vl", "--verbose_lvl", type=int,
        required=False, default=2,
        help=(
            "level of verbose,"
            " default is 2"
        ),
    )
    parser.add_argument(
        "-vs", "--verbose_step", type=int,
        required=False, default=50,
        help=(
            "number of iteration between each display,"
            " default is 50"
        ),
    )

    args = parser.parse_args()
    csv_path = args.file_path  # "data/iris.csv"
    label_col = args.label_fieldname  # "class"
    iterations = args.iterations
    hidden_layers = args.hidden_layers

    functions = {
        'id': identity,
        'exp': ExponentialDecay(args.decay_rate),
        'step': StepFun(lambda x: args.decay_rate * x, 50),
        'inv': InverseDecay(args.learning_rate, args.decay_rate),
    }

    learning_kwargs = {
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'decay_fun': functions[args.decay],

        'batch_size': args.batch_size,
        'iterations': args.iterations,

        'verbose_lvl': args.verbose_lvl,
        'verbose_step': args.verbose_step,
    }
    main(csv_path, label_col, hidden_layers, learning_kwargs)
