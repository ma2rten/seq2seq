import numpy as np

from util import tanh, sigmoid, sigmoid_grad, tanh_grad
from util import initalize, zeros, ones


class Lstm(object):

    def __init__(self, input_size, hidden_size, init_range=1.0, previous=None):
        self.input_size, self.hidden_size = input_size, hidden_size

        if previous:
            self.previous = previous
            previous.next = self

        # initalize weights
        def init(x,y):
            return initalize((x,y), init_range)

        h, n = hidden_size, input_size

        self.W_hi, self.W_hf, self.W_ho, self.W_hj = init(h, h), init(h, h), init(h, h), init(h, h)
        self.W_xi, self.W_xf, self.W_xo, self.W_xj = init(h, n), init(h, n), init(h, n), init(h, n)
        self.b_i, self.b_f, self.b_o, self.b_j = zeros(h), ones(h) * 3, zeros(h), zeros(h)

        # initalize gradients
        self.dW_hi, self.dW_hf, self.dW_ho, self.dW_hj = zeros(h, h), zeros(h, h), zeros(h, h), zeros(h, h)
        self.dW_xi, self.dW_xf, self.dW_xo, self.dW_xj = zeros(h, n), zeros(h, n), zeros(h, n), zeros(h, n)
        self.db_i, self.db_f, self.db_o, self.db_j = zeros(h), zeros(h), zeros(h), zeros(h)

        # list of all parameters
        self.params = [
            ('W_hi', self.W_hi, self.dW_hi),
            ('W_hf', self.W_hf, self.dW_hf),
            ('W_ho', self.W_ho, self.dW_ho),
            ('W_hj', self.W_hj, self.dW_hj),

            ('W_xi', self.W_xi, self.dW_xi),
            ('W_xf', self.W_xf, self.dW_xf),
            ('W_xo', self.W_xo, self.dW_xo),
            ('W_xj', self.W_xj, self.dW_xj),

            ('b_i', self.b_i, self.db_i),
            ('b_f', self.b_f, self.db_f),
            ('b_o', self.b_o, self.db_o),
            ('b_j', self.b_j, self.db_j),
        ]

        self.initSequence()

    def initSequence(self):
        self.t = 0
        self.x = {}
        self.h = {}
        self.c = {}
        self.ct = {}

        self.input_gate = {}
        self.forget_gate = {}
        self.output_gate = {}
        self.cell_update = {}

        if hasattr(self, 'previous'):
            self.h[0] = self.previous.h[self.previous.t]
            self.c[0] = self.previous.c[self.previous.t]
        else:
            self.h[0] = zeros(self.hidden_size)
            self.c[0] = zeros(self.hidden_size)

        if hasattr(self, 'next'):
            self.dh_prev = self.next.dh_prev
            self.dc_prev = self.next.dc_prev
        else:
            self.dh_prev = zeros(self.hidden_size)
            self.dc_prev = zeros(self.hidden_size)

        # reset all gradients to zero
        for name, param, grad in self.params:
            grad[:] = 0

    def forward(self, x_t):
        self.t += 1

        t = self.t
        h = self.h[t-1]

        self.input_gate[t] = sigmoid(np.dot(self.W_hi, h) + np.dot(self.W_xi, x_t) + self.b_i)
        self.forget_gate[t] = sigmoid(np.dot(self.W_hf, h) + np.dot(self.W_xf, x_t) + self.b_f)
        self.output_gate[t] = sigmoid(np.dot(self.W_ho, h) + np.dot(self.W_xo, x_t) + self.b_o)
        self.cell_update[t] = tanh(np.dot(self.W_hj, h) + np.dot(self.W_xj, x_t) + self.b_j)

        self.c[t] = self.input_gate[t] * self.cell_update[t] + self.forget_gate[t] * self.c[t-1]
        self.ct[t] = tanh(self.c[t])
        self.h[t] = self.output_gate[t] * self.ct[t]

        self.x[t] = x_t

        return self.h[t]

    def backward(self, dh):
        t = self.t

        dh = dh + self.dh_prev
        dC = tanh_grad(self.ct[t]) * self.output_gate[t] * dh + self.dc_prev

        d_input = sigmoid_grad(self.input_gate[t]) * self.cell_update[t] * dC
        d_forget = sigmoid_grad(self.forget_gate[t]) * self.c[t-1] * dC
        d_output = sigmoid_grad(self.output_gate[t]) * self.ct[t] * dh
        d_update = tanh_grad(self.cell_update[t]) * self.input_gate[t] * dC

        self.dc_prev = self.forget_gate[t] * dC

        self.db_i += d_input
        self.db_f += d_forget
        self.db_o += d_output
        self.db_j += d_update

        h_in = self.h[t-1]

        self.dW_xi += np.outer(d_input, self.x[t])
        self.dW_xf += np.outer(d_forget, self.x[t])
        self.dW_xo += np.outer(d_output, self.x[t])
        self.dW_xj += np.outer(d_update, self.x[t])

        self.dW_hi += np.outer(d_input, h_in)
        self.dW_hf += np.outer(d_forget, h_in)
        self.dW_ho += np.outer(d_output, h_in)
        self.dW_hj += np.outer(d_update, h_in)

        self.dh_prev = np.dot(self.W_hi.T, d_input)
        self.dh_prev += np.dot(self.W_hf.T, d_forget)
        self.dh_prev += np.dot(self.W_ho.T, d_output)
        self.dh_prev += np.dot(self.W_hj.T, d_update)

        dX = np.dot(self.W_xi.T, d_input)
        dX += np.dot(self.W_xf.T, d_forget)
        dX += np.dot(self.W_xo.T, d_output)
        dX += np.dot(self.W_xj.T, d_update)

        self.t -= 1

        return dX
