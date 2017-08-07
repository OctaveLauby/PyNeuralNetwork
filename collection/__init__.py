from .functions import (
    identity,
    euclidean_dist, euclidean_dist_jac,
    sigmoid, sigmoid_der,
    tanh, tanh_der,
    arctan, arctan_der,
    relu, relu_der,
    softplus, softplus_der,
)
from .function_creators import (
    ExponentialDecay,
    InverseDecay,
    Linear,
    StepFun,
)


ACT_FUN_DER = {
    'sigmoid': (sigmoid, sigmoid_der),
    'tanh': (tanh, tanh_der),
    'arctan': (arctan, arctan_der),
    'relu': (relu, relu_der),
    'softplus': (softplus, softplus_der),
}

COST_FUN_DER = {
    'euclidean': (0.5 * euclidean_dist, 0.5 * euclidean_dist_jac),
}

DECAY_FUN_CREATORS = {
    'id': lambda alpha, k: identity,
    'exp': lambda alpha, k: ExponentialDecay(k),
    'inv': lambda alpha, k: InverseDecay(alpha, k),
    'linear': lambda alpha, k: Linear(k),
}

AVAILABLE_ACT = list(ACT_FUN_DER.keys())
AVAILABLE_COST = list(COST_FUN_DER.keys())
AVAILABLE_DECAY = list(DECAY_FUN_CREATORS.keys())
AVAILABLE_ACT.sort()
AVAILABLE_COST.sort()
AVAILABLE_DECAY.sort()
