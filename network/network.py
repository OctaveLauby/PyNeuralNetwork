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
                output (np.array), expected_output (np.array) -> cost (float)
            cost_jac (callable): derivate of cost_fun (shared by each
                partial derivates)
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

    def fit(self, input_set, output_set, learning_rate, momentum=0,
            batch_size=1, shuffle=True, iterations=1, verbose=True):
        """
        Args:
            input_set (list of np.array[dim_in])
            output_set (list of np.array[dim_out])
            learning_rate (float)
            momentum (float, optional): usually around 0.9
            batch_size (int, optional)
            shuffle (boolean, optional): shuffle data_set when browsing it
            iterations (int, optional): number of data set browsing
        """
        self.use_memory(True)
        assert len(input_set) == len(output_set)
        ds_size = len(input_set)  # size of dataset
        assert ds_size > 0

        for iteration in range(iterations):
            indexes = list(range(ds_size))
            if shuffle:
                random.shuffle(indexes)
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

                if verbose:
                    print(
                        "Batch size : {} | Mean cost : {}"
                        .format(len(batch_indexes, np.mean(costs)))
                    )

    def predict(self, input_set):
        self.use_memory(False)
        return [
            self.forward(vector)
            for vector in input_set
        ]

    @property
    def nL(self):
        return self.nE
