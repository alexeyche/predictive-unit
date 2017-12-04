from util import *
import autograd.numpy as np
from autograd import elementwise_grad as grad
import numpy

np.random.seed(10)

in_n = 3
out_n = 5

batch_size = 10

y = np.random.random((batch_size, out_n))
x = np.random.random((batch_size, in_n))

def f(params):
	W = params.reshape((in_n, out_n))

	# return np.dot(x, W) - np.dot(y, np.dot(W.T, W))

	return np.dot(x - np.dot(y, W.T), W)


	# return - np.dot(y, np.dot(W.T, W))
	# return np.sum(np.dot(y, W.T))

	# return np.sum(np.dot(W, W.T))


df = grad(f)


params = np.random.random((in_n*out_n))
W = params.reshape((in_n, out_n))



o = f(params)

do = df(params).reshape((in_n, out_n))

# do_h = np.outer(y, 2.0*np.dot(y, W.T)).T

# do_h = np.outer(y, np.sum(W, 1)*2.0).T

dW = np.zeros(W.shape)

y_p = np.tile(np.expand_dims(y, 1), (1, out_n, 1))

# x_p = np.tile(np.expand_dims(x, 1), (1, out_n, 1))


dW = np.tile(np.sum(x, 0), (out_n, 1)).T - np.sum(np.dot(y_p + np.transpose(y_p, (0, 2, 1)), W.T), 0).T

print do - dW