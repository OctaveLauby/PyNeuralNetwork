Neural Network with Backpropagation
---

# Introduction

Here is an implementation of a neural network with backpropagation.

The purpose is to implement a **simple and readable** model of what a neural network is, and how it works. The performance is low as it does not us a fully matrix-based approach.

The question of **performance is out of concern**, if you are looking for performance please use appropriate tools.


# Nomenclature

| notation | meaning |
| - | - |
| **l** | index of layer |
| **nl** | index of next layer |
| **j** | index of neuron |
| **k** | index of weight |
| **nN** | number of neurons |
| **nN_l** | number of neurons of layer l |
| **nW** | number of weights |
| **nW_l** | number of weights per neuron on layer l (must be equal) |
| **nW_l_j** | number of weights of neuron j on layer l |

np.array[n] is a numpy array of n elements
2d-np.array[n, m] is a 2d numpy array of n * m elements 


# Sources

- (general) https://ayearofai.com/rohan-lenny-1-neural-networks-the-backpropagation-algorithm-explained-abf4609d4f9d
- (maths) http://neuralnetworksanddeeplearning.com/chap2.html
- (hyperparameters) http://cs231n.github.io/neural-networks-3/
- (dataset) https://archive.ics.uci.edu/ml/datasets/Iris
