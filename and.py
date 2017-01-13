#!/usr/bin/env python
# https://www.analyticsvidhya.com/blog/2016/04/neural-networks-python-theano/
import theano
import theano.tensor as T
import numpy as np
from random import random
import matplotlib.pyplot as plt


# input
x = T.matrix('x')
w = theano.shared(np.array([random(), random()]))
b = theano.shared(1.)
learning_rate = 0.01
a_hat = T.vector() # actual output

# output
z = T.dot(x, w) + b
a = 1 / (1 + T.exp(-z)) # activation function: sigmoid
cost = -(a_hat*T.log(a) + (1 - a_hat)*T.log(1-a)).sum()

# training function: gradient descent
dw, db = T.grad(cost, [w, b])
train = theano.function(
    inputs = [x, a_hat],
    outputs = [a, cost],
    updates = [
        (w, w - learning_rate*dw),
        (b, b - learning_rate*db)
    ]
)

######################################################

# inputs and outputs
inputs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]
outputs = [
    0,
    0,
    0,
    1
]

# loop over (inputs, outputs) & update (weights, bias)
cost = []
for iteration in range(30000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)

# print the outputs
for i in range(len(inputs)):
    print('%d AND %d = %f' % (inputs[i][0], inputs[i][1], pred[i]))

# plot the cost
plt.plot(cost)
plt.show()

