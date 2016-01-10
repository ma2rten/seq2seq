import numpy as np
from numpy.random import rand


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return x * (1.0 - x)


def tanh_grad(x):
    return 1 - x ** 2


def initalize(dim, init_range):
    return rand(*dim) * init_range


def zeros(*dim):
    return np.zeros(dim)


def ones(*dim):
    return np.ones(dim)
