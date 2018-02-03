from util import *
import autograd.numpy as np
from autograd import elementwise_grad as grad
import numpy


in_n = 3
out_n = 5


y = np.random.random(out_n)

def f(params):
	W = params.reshape((in_n, out_n))

	return np.sum(np.dot(y, np.dot(W.T, W)))
	# return np.sum(np.dot(y, W.T))

	# return np.sum(np.dot(W, W.T))


df = grad(f)


params = np.random.random((in_n*out_n))
W = params.reshape((in_n, out_n))



o = f(params)

do = df(params).reshape((in_n, out_n))

# do_h = np.outer(y, 2.0*np.dot(y, W.T)).T

# do_h = np.outer(y, np.sum(W, 1)*2.0).T

print np.dot(np.asarray([y]*out_n) + np.asarray([y]*out_n).T, W.T) - do.T

# np.dot(np.expand_dims(y, 1), np.expand_dims(np.dot(y, W.T), 0))

# outer = np.dot(np.expand_dims(y, 1), np.expand_dims(y, 0))


# do_h = np.dot(y, W.T).T
# do_h = np.dot(2.0*outer, W.T).T




# do_h = np.outer(y, np.dot(y, np.dot(W.T, W)))


# from sympy import *

# w0, w1, w2, w3 = symbols("w0 w1 w2 w3")


# w = Matrix(2, 2, [w0, w1, w2, w3])
