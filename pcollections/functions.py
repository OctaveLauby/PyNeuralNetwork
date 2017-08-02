import numpy as np

from utils.function import Function

euclidean_dist = Function(lambda v1, v2: np.linalg.norm(v1 - v2))
euclidean_dist_jac = Function(lambda v1, v2: 2 * (v1 - v2))
