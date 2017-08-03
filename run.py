import numpy as np
import random
from csv import DictReader
from pprint import pprint

from pcollections.functions import (
    euclidean_dist, euclidean_dist_jac,
    sigmoid, sigmoid_der,
)
from network import NNetwork, HNLayer


# --------------------------------------------------------------------------- #
# DataSet reading

path = "iris.csv"
vector_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
label_col = "class"

input_set = []
input_labels = []

with open(path) as csvfile:
    reader = DictReader(csvfile)
    for row in reader:
        input_set.append(np.array([
            float(row[col])
            for col in vector_cols
        ]))
        input_labels.append(row[label_col])


def label2vector(label):
    if label == "Iris-setosa":
        return np.array([1, 0, 0])
    elif label == "Iris-versicolor":
        return np.array([0, 1, 0])
    elif label == "Iris-virginica":
        return np.array([0, 0, 1])
    else:
        raise Exception("Unknown Label '%s'" % label)


def vector2label(vector):
    if max(vector) == vector[0]:
        return "Iris-setosa"
    elif max(vector) == vector[1]:
        return "Iris-versicolor"
    elif max(vector) == vector[2]:
        return "Iris-virginica"


def prediction2labels(prediction):
    return [
        vector2label(vector)
        for vector in prediction
    ]


def rights_ratio(labels, prediction):
    return float(sum([
        label == pred
        for label, pred in zip(labels, prediction2labels(prediction))
    ]) / len(prediction))


output_set = [label2vector(label) for label in input_labels]


# --------------------------------------------------------------------------- #
# Initialization

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
learning_kwargs = {
    'learning_rate': 0.001,
    'momentum': 0.9,

    'batch_size': 5,
    'iterations': 1000,

    'verbose_lvl': 2,
    'verbose_step': 50,
}

# Training
network.fit(input_set, output_set, **learning_kwargs)

# Test
prediction = network.predict(input_set)
print(
    "Rights : %.1f %%"
    % (100 * rights_ratio(input_labels, prediction))
)
