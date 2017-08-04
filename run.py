# --------------------------------------------------------------------------- #
# DataSet reading

from utils.dataset import DataSet

csv_path = "iris.csv"
vector_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
label_col = "class"

ds = DataSet(csv_path, vector_cols, label_col)
input_set = ds.input_set
input_labels = ds.input_labels
output_set = ds.output_set

# ds.display()


# --------------------------------------------------------------------------- #
# Initialization
from network import HNN

# Network
network = HNN(
    dim_in=ds.dim_in,
    dim_out=ds.dim_out,
    hidden_layers_nN=[3],
)

print("**** Original Network :")
network.pprint()
print()


# --------------------------------------------------------------------------- #
# Learning
from pcollections.function_creators import (
    exponential_decay,
    step_decay,
    inverse_decay,
)

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
