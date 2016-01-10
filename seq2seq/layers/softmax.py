import numpy as np
from util import initalize


class Softmax (object):
    def __init__(self, input_size, output_size, init_range=1.0):
        self.input_size = input_size
        self.output_size = output_size

        self.W = initalize((output_size, input_size), init_range)
        self.dW = np.zeros((output_size, input_size))

        self.params = [
            ('W', self.W, self.dW)
        ]

    def initSequence(self):
        self.pred = []
        self.x = []
        self.targets = []
        self.t = 0
        self.dW[:] = 0

    def forward(self, x):
        self.t += 1

        y = self.W.dot(x)
        y = np.exp(y - y.max())  # for numerical stability
        y /= y.sum()

        self.pred.append(y)
        self.x.append(x)

        return y

    def backward(self, target):
        self.t -= 1
        self.targets.append(target)

        x = self.x[self.t]
        d = self.pred[self.t].copy()
        d[target] -= 1

        self.dW += np.outer(d, x)
        delta = np.dot(self.W.T, d)

        return delta

    def getCost(self):
        return sum(-np.log(y[target]) for target, y in zip(self.targets, reversed(self.pred)))
