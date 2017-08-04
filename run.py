from network import HNN
from pcollections.function_creators import (
    exponential_decay,
    step_decay,
    inverse_decay,
)
from utils.dataset import DataSet


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
    csv_path = "data/iris.csv"
    label_col = "class"
    hidden_layers = 1
    learning_kwargs = {
        'learning_rate': 0.001,
        'momentum': 0.9,

        # 'learning_rate': 0.1,
        # 'momentum': 0.9,
        # 'decay_fun': step_decay(25, 0.5),
        # 'decay_fun': exponential_decay(1),

        'batch_size': 1,
        'iterations': 300,

        'verbose_lvl': 2,
        'verbose_step': 50,
    }
    main(csv_path, label_col, hidden_layers, learning_kwargs)
