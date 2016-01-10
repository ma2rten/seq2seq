import numpy as np
from numpy.random import randn
from random import randint

from layers import Lstm, Softmax, Embedding

DELTA = 1e-5
THRESHOLD = 1e-2

EOS = 0
HIDDEN_SIZE = 10

input_layers = [
    Embedding(5, 10),
    Lstm(10, 10),
]

output_layers = [
    Embedding(5, 10),
    Lstm(10, 10, previous=input_layers[1]),
    Softmax(10, 4),
]

X = [randint(0, 4), randint(0, 4)]
Y = [randint(0, 3), randint(0, 3)]


def train():
    # reset state
    for layer in input_layers:
        layer.initSequence()

    # forward
    for x in X:
        h = x
        for layer in input_layers:
            h = layer.forward(h)

    # reset state
    for layer in output_layers:
        layer.initSequence()

    for y in [EOS] + Y:
        h = y
        for layer in output_layers:
            h = layer.forward(h)

    # backwards
    for y in reversed(Y + [EOS]):
        delta = y
        for layer in reversed(output_layers):
            delta = layer.backward(delta)

    for x in reversed(X):
        delta = np.zeros(HIDDEN_SIZE)
        for layer in reversed(input_layers):
            delta = layer.backward(delta)

    return output_layers[-1].getCost()


def numeric_gradient(mat, dmat, i):
    val = mat.flat[i]

    mat.flat[i] = val + DELTA
    loss1 = train()
    mat.flat[i] = val - DELTA
    loss2 = train()

    mat.flat[i] = val

    return (loss1 - loss2) / (2 * DELTA)


def main():
    passed = True

    for layer in input_layers + output_layers:
        for name, mat, dmat in layer.params:
            for i in xrange(mat.size):
                grad_num = numeric_gradient(mat, dmat, i)
                grad_analytic = dmat.flat[i]

                if grad_num == 0 and grad_analytic == 0:
                    continue

                error = abs(grad_analytic - grad_num)

                if error > THRESHOLD or np.isnan(error):
                    print layer, name, "ERROR", grad_analytic, grad_num
                    passed = False
                    break

    if passed:
        print "ALL PASSED"

if __name__ == "__main__":
    main()
