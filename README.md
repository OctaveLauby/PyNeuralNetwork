Neural Network with Backpropagation
---

# Introduction

Here is an implementation of a neural network with backpropagation.

The purpose is to implement a **simple and readable** model of what a neural network is, and how it works. The performance is low as it does not us a fully matrix-based approach.

The question of **performance is out of concern**, if one is looking for performance please use appropriate tools.


# Use Case

## What one will use

**Script**: *run.py*

**DataSet**: example are those in *data/*

- ***csv*** file
- with ***header***
- ***','*** delimiter
- ***number*** columns, except for label columns that can be anything

**About**: Here are some things one does not need to handle:

- ***Input space dimension*** is determined by the number of columns (- label column)
- ***Output space dimension*** is determined by the number of labels
- Input layer and output layer size match input and output dim
- Data set is ***splitted*** in 2 (80%, 20%) to build a training set and a globalisation set. ***Proportions*** of labels are respected.
- Training set is ***shuffled*** at each iteration.


## How one should use it

1. See all options:

`python3 run.py -h`

2. Pick your dataset: *-f -l -s*
> We will use data/iris.csv, where label is given in in 'class' column. We can use the input normalisation (-s option) 

`python3 run.py -f data/iris.csv -l class -s`


3. Customize DNN: *-n --hidden_act --output_act*.
> For instance lets work with 2 hidden layers with arctan activation function, and an output layer with tanh function

`python3 run.py -f data/iris.csv -l class -n 2 --hidden_act arctan --output_act tanh `

4. Customize weights update: *-r -m*
> For instance lets use a learning rate of 0.01 and a momentum of 0.8

`python3 run.py f data/iris.csv -l class -r 0.01 -m 0.8`

5. Customize learning rate update: *--decay --decay_rate --decay_step*
> For instance lets update learning rate every 30 iterations, using exp(-ki) decay funcion where k = 0.1

`python3 run.py f data/iris.csv -l class --decay exp --decay_rate 0.1 --decay_step 30`

6. Customize learning loop: *-i -b*
> For instance lets loop 200 times on training set

`python3 run.py f data/iris.csv -l class --decay exp --decay_rate 0.1 --decay_step 30`

7. Customize verbose: *--verbose_lvl --verbose_step*
> For instance lets display information per batch every 50 step
> verbose_lvl = 0 for no display, 1 for start and end, 2 for iteration level, 3 for batch level

`python3 run.py f data/iris.csv -l class --verbose_lvl 3 --verbose_step 50`

8. Combinaison of those
> My prefered setting is basically the default one (+ the -s option to rescale input)
> But one can combine all the options the way one wants


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


# Todo

- Weight regulation
- KeyInterruption -> set a new lambda or sth
- Smart lambda decay (when cost is not moving for a while)
- User can set layer sizes


# Leads

- First layer bigger than output might work better
- Sigmoid activation on output can cause issues
- Biases are  typically initialized to 0
- Tanh squashing function on the output ? -> stronger gradiant


# Sources

- (general) https://ayearofai.com/rohan-lenny-1-neural-networks-the-backpropagation-algorithm-explained-abf4609d4f9d
- (maths) http://neuralnetworksanddeeplearning.com/chap2.html
- (hyperparameters) http://cs231n.github.io/neural-networks-3/
- (hyperparameters) http://colinraffel.com/wiki/neural_network_hyperparameters
- (dataset) https://archive.ics.uci.edu/ml/datasets.html
- (number of layers and nodes) http://dstath.users.uth.gr/papers/IJRS2009_Stathakis.pdf