
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *

np.random.seed(34) 



act = BoundRelu()


batch_size = 1
layer_size = 1

input_size = 3

t_seq = 100
tau = 2.0
alpha_m = 0.99

# x_t = 0.1*np.random.randn(t_seq, batch_size, input_size)
# x_t = np.random.random((t_seq, batch_size, input_size))
# x_t = 0.5*(x_t < 0.05).astype(np.float32)

x_t = np.zeros((t_seq, batch_size, input_size))
x_t[25,0,0] = 1.0
x_t[50,0,1] = 1.0
x_t[75,0,2] = 1.0


x_t = smooth_batch_matrix(x_t, kernel=exp_filter)

# a_t = np.random.random((t_seq, batch_size, layer_size))
# a_t = (a_t < 0.05).astype(np.float32)


a_t_target = np.zeros((t_seq, batch_size, layer_size))
a_t_target[51,0,0] = 1.0
a_t_target = smooth_batch_matrix(a_t_target, kernel=exp_filter)

# a_t = smooth_batch_matrix(a_t)

W = 0.1 - 0.2*np.random.random((input_size, layer_size))
b = np.zeros((layer_size,))


W = norm(W)


lrate = 0.5
fb_factor = 1.0


def predictive_dynamics(x, u, a, a_target, fb_factor):
	# e = x - np.dot(a, W.T)
	# du = np.dot(e, W) - u + fb_factor * (a_target - a)
	# u_new = u + du / tau

	e = x - np.dot(a, W.T)
	du = np.dot(e, W) - u
	
	u_new = u + du / tau
	a_new = act(u_new)
	u_new = u_new + fb_factor * (a_target - a_new) / tau
	return u_new, act(u_new)

def simple_dynamics(x, u, a, a_target, fb_factor):
	du = np.dot(x, W) - u
	u_new = u + du / tau
	a_new = act(u_new)
	u_new = u_new + fb_factor * (a_target - a_new) / tau 
	return u_new, act(u_new) 


def no_dynamics(x, u, a, a_target, fb_factor):
	u_new = np.dot(x_t[ti], W) 
	u_new += fb_factor * (a_t_target[ti] - act(u_new))
	# u_new = np.dot(x, W) + fb_factor * (a_target - a)
	return u_new, act(u_new) 

dynamics = no_dynamics


epochs = 1000
ehh = np.zeros((epochs, t_seq, batch_size, 1))
am = np.zeros((batch_size, layer_size))
edWh = np.zeros((epochs, t_seq, input_size, layer_size))
edW = np.zeros((epochs, input_size, layer_size))
eavh = np.zeros((epochs, t_seq, batch_size, layer_size))

for epoch in xrange(epochs):
	u = np.zeros((batch_size, layer_size))
	a = np.zeros((batch_size, layer_size))

	uh = np.zeros((t_seq, batch_size, layer_size))
	ah = np.zeros((t_seq, batch_size, layer_size))
	amh = np.zeros((t_seq, batch_size, layer_size))
	eh = np.zeros((t_seq, batch_size, 1))
	dh = np.zeros((t_seq, batch_size, layer_size))

	dWacc = np.zeros((input_size, layer_size))
	dWh = np.zeros((t_seq, input_size, layer_size))
	
	for ti in xrange(t_seq):
		u, a = dynamics(x_t[ti], u, a, a_t_target[ti], fb_factor)
		
		e = np.linalg.norm(a_t_target[ti] - a)

		# dW = np.dot(x_t[ti].T, a_t_target[ti] - a)
		
		# dW = np.dot(x_t[ti].T, a - 0.1)
		dW = np.dot(x_t[ti].T, a*np.sign(a-am))
		# dW = np.dot((x_t[ti] - np.dot(a, W.T)).T, a*np.sign(a-0.1))

		# dW = np.dot((x_t[ti] - np.dot(a, W.T)).T, a - 0.05)

		dWacc += dW/t_seq

		dWh[ti] = dW.copy()
		uh[ti] = u.copy()
		ah[ti] = a.copy()
		eh[ti] = e
		dh[ti] = a_t_target[ti]	- a

		am = (1.0 - alpha_m) * a + alpha_m * am

		amh[ti] = am.copy()

	ehh[epoch] = eh.copy()
	edWh[epoch] = dWh.copy()
	edW[epoch] = dWacc.copy()

	uv = np.zeros((batch_size, layer_size))
	av = np.zeros((batch_size, layer_size))

	uvh = np.zeros((t_seq, batch_size, layer_size))
	avh = np.zeros((t_seq, batch_size, layer_size))

	for ti in xrange(t_seq):
		uv, av = dynamics(x_t[ti], uv, av, a_t_target[ti], 0.0)

		uvh[ti] = uv.copy()
		avh[ti] = av.copy()

	eavh[epoch] = avh.copy()
	W += lrate * dWacc
	W = norm(W)

	train_error = np.sum((ah - a_t_target) ** 2)
	valid_error = np.sum((avh - a_t_target) ** 2)
	if epoch % 2 == 0:
		print "Epoch {} train_error {} valid_error {}".format(epoch, train_error, valid_error)


# shl(ehh[0], ehh[epochs/2], ehh[-1])