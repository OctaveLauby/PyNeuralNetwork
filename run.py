"""Run a network training on a dataset.

Examples in README
"""

import argparse

from network import HNN
from collection import AVAILABLE_ACT, AVAILABLE_COST, AVAILABLE_DECAY
from utils.dataset import DataSet


def identity(x):
    return x


def main(input_kwargs, network_kwargs, learning_kwargs):
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
        build_params=network_kwargs
    )

    print("**** Original Network :")
    network.display_params()
    network.pprint()
    print()

    # ----------------------------------------------------------------------- #
    # Learning

    # Training
    try:
        network.fit(
            training_set.input_set,
            training_set.output_set,
            **learning_kwargs
        )
    except KeyboardInterrupt:
        print()
        print("/!\\ Fitting stopped. /!\\")
        print()

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
    parser.add_argument(
        "-s", "--std_scale", type=str,
        required=True,
        help=(
            "rescale input with z-score normalization."
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
    parser.add_argument(
        "--hidden_act", choices=AVAILABLE_ACT,
        required=False, default='sigmoid',
        help=(
            "activation function of hidden layers,"
            "default is sigmoid."
        )
    )
    parser.add_argument(
        "--output_act", choices=AVAILABLE_ACT,
        required=False, default='tanh',
        help=(
            "activation function of output layers,"
            "default is tanh."
        )
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
        "--decay", choices=AVAILABLE_DECAY,
        required=False, default='id',
        help=(
            "decay function, set k for decay rate,"
            " default is 'id'"
        ),
    )
    parser.add_argument(
        "--decay_rate", type=float,
        required=False, default=None,
        help=(
            "decay function rate,"
            " default is None"
        ),
    )
    parser.add_argument(
        "--decay_step", type=int,
        required=False, default=1,
        help=(
            "decay function rate,"
            " default is 1"
        ),
    )

    # ---- Verbose
    parser.add_argument(
        "--verbose_lvl", type=int,
        required=False, default=2,
        help=(
            "level of verbose,"
            " default is 2"
        ),
    )
    parser.add_argument(
        "--verbose_step", type=int,
        required=False, default=50,
        help=(
            "number of iteration between each display,"
            " default is 50"
        ),
    )

    args = parser.parse_args()

    input_kwargs = {
        'csv_path': args.file_path,
        'label_col': args.label_fieldname,
        'std_der': args.std_der,
    }
    network_kwargs = {
        'cost_fun': 'euclidean',
        'nHL': args.hidden_layers,
        'act_fun': args.hidden_act,
        'outact_fun': args.output_act,
    }

    learning_kwargs = {
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,

        'decay_fun': args.decay,
        'decay_rate': args.decay_rate,
        'decay_step': args.decay_step,

        'batch_size': args.batch_size,
        'iterations': args.iterations,

        'verbose_lvl': args.verbose_lvl,
        'verbose_step': args.verbose_step,
    }
    main(input_kwargs, network_kwargs, learning_kwargs)
