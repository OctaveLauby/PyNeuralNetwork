import numpy as np
import random

from .base import NContainer
from .layer import NLayer


class NNetwork(NContainer):
    """Neural Network using backpropagation algorithm."""

    def __init__(self, dim_in, dim_out, cost_fun, cost_jac):
        """Create a neural network.

        Args:
            dim_in (int): dimension of input space
            dim_out (int): dimension of output space
            cost_fun (callable): function used to calculate cost
                | output (np.array), expected_output (np.array) -> cost (float)
            cost_jac (callable): jacobian of cost_fun
        """
        super().__init__(dim_in=dim_in, dim_out=dim_out, elem_cls=NLayer)
        self._cost_fun = cost_fun
        self._cost_jac = cost_jac

    def check(self):
        """Make sure dimension chain is consistent."""
        expected_dim = self.dim_in
        for layer in self.iter():
            assert layer.dim_in == expected_dim
            layer.check()
            expected_dim = layer.dim_out
        assert self.dim_out == expected_dim
        return True

    def cost(self, expected_output, output=None):
        if output is None:
            output = self.read_memory('output', last=True)
        return self._cost_fun(output, expected_output)

    # Major steps

    def backward(self, expected_output):
        """STEP-2: Back-propagate error on last output."""
        output = self.read_memory('output', last=True)
        nl_delta = self._cost_jac(output, expected_output)
        nl_weights = 1
        for layer in self.riter():
            nl_delta = layer.backward(
                nl_weights=nl_weights,
                nl_delta=nl_delta,
            )
            nl_weights = layer.weights()

    def forward(self, vector):
        """STEP-1: Forward input vector to calculate output."""
        self.memorize('input', vector)
        int_vect = vector
        for layer in self.iter():
            int_vect = layer.forward(vector=int_vect)
        output = int_vect
        self.memorize('output', output)
        return output

    def update(self, learning_rate, momentum=0):
        """STEP-3: Update weights using last batch of inputs.

        Args:
            learning_rate (float)
            momentum (float): usually around 0.9
        """
        for layer in self.iter():
            layer.update(learning_rate, momentum)
        self.reset_memory()

    # Learning

    def evaluate(self, input_set, output_set):
        outputs = self.predict(input_set)
        return [
            self.cost(expected_output, output=output)
            for output, expected_output in zip(outputs, output_set)
        ]

    def fit(self, input_set, output_set,
            learning_rate=0.001, momentum=0, decay_fun=None,  # update params
            batch_size=1, iterations=1, shuffle=True,  # loop params
            verbose_lvl=1, verbose_step=1,
            ):
        """
        Args:
            input_set (list of np.array[dim_in]):   list of input vectors
            output_set (list of np.array[dim_out]): associated outputs

            learning_rate (float):  learning rate
            momentum (float):       learning momentum, usually around 0.9
            decay_fun (call):       function to update learning_rate
                | float -> float
                > default is identity

            batch_size (int):   size of input batch between each update
            iterations (int):   browsing iterations on data
            shuffle (boolean):  shuffle data_set when browsing it

            verbose_lvl (int):  the higher, the more it display
                0 for None
                1 for start and end display
                2 for iteration display
                3 for batch display
            verbose_step (int): authorize display every x steps
        """
        decay = decay_fun
        if decay_fun is None:
            def decay(x):
                return x
        self.reset_memory()
        assert len(input_set) == len(output_set)
        ds_size = len(input_set)  # size of dataset
        assert ds_size > 0

        if verbose_lvl:
            print("# ---- Fitting on a data set of %s inputs:" % ds_size)
            print("\t learning_rate     %s" % learning_rate)
            print("\t momentum          %s" % momentum)
            print("\t decay_fun         %s" % decay_fun)
            print("\t batch_size        %s" % batch_size)
            print("\t iterations        %s" % iterations)
            print("\t shuffle           %s" % shuffle)
            print()

        iteration = 0
        batch_costs = []
        for iteration in range(1, iterations+1):
            indexes = list(range(ds_size))
            if shuffle:
                random.shuffle(indexes)
            batch_costs = []
            while indexes:
                # create batch
                batch_indexes = []
                while indexes and len(batch_indexes) < batch_size:
                    batch_indexes.append(indexes.pop())

                # learn on batch
                costs = []
                for index in batch_indexes:
                    self.forward(input_set[index])
                    costs.append(self.cost(output_set[index]))
                    self.backward(output_set[index])
                self.update(learning_rate=learning_rate, momentum=momentum)

                batch_cost = np.mean(costs)
                batch_costs.append(batch_cost)
                if verbose_lvl >= 3 and (iteration % verbose_step == 0):
                    print(
                        "\t> Batch size : {} | Mean cost : {}"
                        .format(len(batch_indexes), batch_cost)
                    )

            if iteration is 1 and verbose_lvl:
                print(
                    "# Iteration {} | Mean cost : {} (l={})"
                    .format(1, np.mean(batch_costs), learning_rate)
                )
            if verbose_lvl >= 2 and (iteration % verbose_step == 0):
                print(
                    "> Iteration {} | Mean cost : {} (l={})"
                    .format(iteration, np.mean(batch_costs), learning_rate)
                )

            # End of iteration
            learning_rate = decay(learning_rate)
        if verbose_lvl:
            print(
                "# Iteration {} | Mean cost : {} (l={})"
                .format(iteration, np.mean(batch_costs), learning_rate)
            )

    def predict(self, input_set):
        res = [
            self.forward(vector)
            for vector in input_set
        ]
        self.reset_memory()
        return res

    @property
    def nL(self):
        return self.nE
