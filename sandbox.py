from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier

from utils.dataset import DataSet


ds = DataSet("data/iris.csv", label_col="class")
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


network = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(2, 4),
    random_state=1,
)


network.fit(training_set.input_set, training_set.output_set)


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

print(confusion_matrix(gene_set.output_set, predictions))

print(classification_report(gene_set.output_set, predictions))
