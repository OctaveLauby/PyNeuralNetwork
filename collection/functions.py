import numpy as np

from utils.function import Function

euclidean_dist = Function(lambda var, const: np.linalg.norm(var - const))
euclidean_dist_jac = Function(lambda var, const: 2 * (var - const))

sigmoid = Function(lambda z: 1 / (1 + np.exp(-z)))
sigmoid_der = Function(lambda z: sigmoid(z) * (1 - sigmoid(z)))

tanh = Function(np.tanh)
tanh_der = Function(lambda z: 1 - tanh(z)**2)

arctan = Function(np.arctan)
arctan_der = Function(lambda z: 1 / (z**2 + 1))
