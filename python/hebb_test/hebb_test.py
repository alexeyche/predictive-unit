
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *

np.random.seed(31) 



act = BoundRelu()


batch_size = 1
layer_size = 10

input_size = 100

t_seq = 100
tau = 2.0

lrate = 0.1

# x = np.random.randn(t_seq, batch_size, input_size)

x = np.zeros((t_seq, batch_size, input_size))
for ti in xrange(0, t_seq, 5):
	x[ti, 0, ti % input_size] = 1.0	

# a_t = np.random.random((t_seq, batch_size, layer_size))
# a_t = (a_t < 0.05).astype(np.float32)

a_t = np.zeros((t_seq, batch_size, layer_size))
for ti in xrange(0, t_seq, t_seq/layer_size):
	a_t[ti, 0, ti/layer_size] = 1.0

# a_t = smooth_batch_matrix(a_t)

W = 0.1 - 0.2*np.random.random((input_size, layer_size))
b = np.zeros((layer_size,))




for epoch in xrange(2000):
	u = np.zeros((batch_size, layer_size))
	a = np.zeros((batch_size, layer_size))

	uh = np.zeros((t_seq, batch_size, layer_size))
	ah = np.zeros((t_seq, batch_size, layer_size))
	eh = np.zeros((t_seq, batch_size, input_size))

	dWacc = np.zeros((input_size, layer_size))
	dWh = np.zeros((t_seq, input_size, layer_size))
	for ti in xrange(t_seq):

		# u = np.dot(x[ti], W)
		# u += (np.dot(x[ti], W) - a) * tau
		# u += (np.dot(x[ti] - np.dot(a, W.T), W) - a) * tau

		# u = np.dot(x[ti], W) + (a_t[ti]-a)   # works with 0.5
		# u += (np.dot(x[ti], W) - a + (a_t[ti]-a) ) * tau
		
		e = x[ti] - np.dot(a, W.T)
		u += (np.dot(e, W) + (a_t[ti]-a) ) * tau

		a = act(u)

		# dW = np.dot(x[ti].T, a_t[ti] - a)
		dW = np.dot(x[ti].T, a - 0.1)
		# dW = np.dot((x[ti] - np.dot(a, W.T)).T, a-0.1)

		dWacc += dW/t_seq

		dWh[ti] = dW.copy()
		uh[ti] = u.copy()
		ah[ti] = a.copy()
		eh[ti] = e.copy()

	W += lrate * dWacc

	# shl(dWacc[:,0])
	
	# shl(dWacc[:,7]) # ??

	error = np.sum((ah - a_t) ** 2)
	if epoch % 50 == 0:
		print "Epoch {} error {}".format(epoch, error)


