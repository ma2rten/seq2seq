import numpy as np
from layers import Lstm, Softmax, Embedding
from numpy.random import randint

EOS = 0

HIDDEN_SIZE = 10
EMBED_SIZE = 10
INPUT_SIZE = 4
OUTPUT_SIZE = 4
INIT_RANGE = 1.0

LEARNING_RATE = 0.7
CLIP_GRAD = 5.0


class Seq2seq (object):

    def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE, embed_size=EMBED_SIZE, lr=LEARNING_RATE, clip_grad=CLIP_GRAD, init_range=INIT_RANGE):
        # this model will generate a vector representation based on the input
        input_layers = [
            Embedding(input_size, embed_size, init_range),
            Lstm(embed_size, hidden_size, init_range),
            #Lstm(hidden_size, hidden_size),
        ]

        # this model will generate an output sequence based on the hidden vector
        output_layers = [
            Embedding(output_size, embed_size, init_range),
            Lstm(embed_size, hidden_size, init_range, previous=input_layers[1]),
            #Lstm(hidden_size, hidden_size, previous=input_layers[2]),
            Softmax(hidden_size, output_size, init_range)
        ]

        self.input_layers, self.output_layers = input_layers, output_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.clip_grad = clip_grad

    def predict(self, X, max_length=10):
        # reset state
        for layer in self.input_layers:
            layer.initSequence()

        # process input sequence
        for x in X:
            h = x
            for layer in self.input_layers:
                h = layer.forward(h)

        # reset state
        for layer in self.output_layers:
            layer.initSequence()

        # apply output model
        out = []
        token = EOS

        while len(out) < max_length:
            # start with last generated token
            h = token

            # go though all layers
            for layer in self.output_layers:
                h = layer.forward(h)

            # select token with highest softmax activation
            token = np.argmax(h)

            # stop if we generated end of sequence token
            if token == EOS:
                break

            # add token to output sequence
            out.append(token)

        return out

    def train(self, X, Y):
        # reset state
        for layer in self.input_layers:
            layer.initSequence()

        # forward pass
        for x in X:
            h = x
            for layer in self.input_layers:
                h = layer.forward(h)

        # reset state
        for layer in self.output_layers:
            layer.initSequence()

        for y in [EOS] + Y:
            h = y
            for layer in self.output_layers:
                h = layer.forward(h)

        # backward pass
        for y in reversed(Y + [EOS]):
            delta = y
            for layer in reversed(self.output_layers):
                delta = layer.backward(delta)

        for x in reversed(X):
            delta = np.zeros(self.hidden_size)
            for layer in reversed(self.input_layers):
                delta = layer.backward(delta)

        # gradient clipping
        grad_norm = 0.0

        for layer in self.input_layers + self.output_layers:
            for name, param, grad in layer.params:
                grad_norm += (grad ** 2).sum()

        grad_norm = np.sqrt(grad_norm)

        # sgd
        for layer in self.input_layers + self.output_layers:
            for name, param, grad in layer.params:
                if grad_norm > self.clip_grad:
                    grad /= grad_norm / self.clip_grad
                param -= self.lr * grad

        return self.output_layers[-1].getCost()
