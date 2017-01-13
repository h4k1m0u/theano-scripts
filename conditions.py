#!/usr/bin/env python
import theano.tensor as T
from theano.ifelse import ifelse
import theano
import numpy as np

a, b = T.scalars('a', 'b')
z = ifelse(T.gt(a, b), a, b)
f = theano.function([a, b], z)

print(f(1.0, 2.0))
print(f(2.0, 1.0))
