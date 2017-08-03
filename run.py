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

ds.display()


# --------------------------------------------------------------------------- #
# Initialization
import random
from network import NNetwork, HNLayer
from pcollections.functions import (
    euclidean_dist, euclidean_dist_jac,
    sigmoid, sigmoid_der,
)

# Network
cost_fun = euclidean_dist
cost_jac = euclidean_dist_jac
network = NNetwork(dim_in=4, dim_out=3, cost_fun=cost_fun, cost_jac=cost_jac)

# Neurons
n_kwargs = {
    'act_fun': sigmoid,
    'act_der': sigmoid_der,
    'init_fun': lambda i: random.random(),
}

# Layers
network.add(HNLayer(dim_in=4, nN=3, **n_kwargs))
network.add(HNLayer(dim_in=3, nN=3, **n_kwargs))

network.check()
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
    'learning_rate': 0.1,
    'momentum': 0.9,
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
