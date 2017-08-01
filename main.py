from network import NNetwork, HNLayer



def cost_fun(*args, **kwargs):
    raise NotImplementedError


def act_fun(value):
    return value


def act_der(value):
    return 1


def init_fun():
    return 0.1


network = NNetwork(dim_in=3, dim_out=2, cost_fun=cost_fun)

n_kwargs = {
    'dim_in': 3,
    'act_fun': act_fun,
    'act_der': act_der,
    'init_fun': init_fun,
}
layer_1 = HNLayer(dim_in=3, nN=3, n_kwargs=n_kwargs)
layer_2 = HNLayer(dim_in=3, nN=2, n_kwargs=n_kwargs)

network.add(layer_1)
network.add(layer_2)

network.pprint()
