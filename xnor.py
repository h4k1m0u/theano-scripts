#!/usr/bin/env python
# https://www.analyticsvidhya.com/blog/2016/04/neural-networks-python-theano/
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np
from random import random


# define variables
x = T.matrix('x')
w1 = theano.shared(np.array([random(), random()]))
w2 = theano.shared(np.array([random(), random()]))
w3 = theano.shared(np.array([random(), random()]))
b1 = theano.shared(1.)
b2 = theano.shared(1.)
learning_rate = 0.01

# define mathematical expressions
a1 = 1 / (1 + T.exp(-T.dot(x, w1) - b1))
a2 = 1 / (1 + T.exp(-T.dot(x, w2) - b1))
x2 = T.stack([a1, a2], axis=1)
a3 = 1 / (1 + T.exp(-T.dot(x2, w3) - b2))

# define gradient and update rule
a_hat = T.vector('a_hat') # actual output
cost = -(a_hat*T.log(a3) + (1 - a_hat)*T.log(1-a3)).sum()
dw1, dw2, dw3, db1, db2 = T.grad(cost, [w1, w2, w3, b1, b2])
train = theano.function(
    inputs = [x, a_hat],
    outputs = [a3, cost],
    updates = [
        [w1, w1 - learning_rate*dw1],
        [w2, w2 - learning_rate*dw2],
        [w3, w3 - learning_rate*dw3],
        [b1, b1 - learning_rate*db1],
        [b2, b2 - learning_rate*db2]
    ]
)

# train the model
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [1, 0, 0, 1]

cost = []
for _ in range(30000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)

# print the outputs
for i in range(len(inputs)):
    print('%d XNOR %d = %d' % (inputs[i][0], inputs[i][1], pred[i]))
