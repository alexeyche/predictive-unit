
from util import *
import autograd.numpy as np
from autograd import elementwise_grad as grad
import numpy

input_size = 20
net_size = 10
output_size = 3

np.random.seed(11)

def slice_params(params):
	idx = (input_size*net_size, net_size*output_size)

	W0 = params[:idx[0]].reshape((input_size, net_size))
	W1 = params[idx[0]:(idx[0]+idx[1])].reshape((net_size, output_size))
	
	return W0, W1


x = np.random.random(input_size)


def diff_to_constant(v0, v1):
	g = np.where(v1 != 0.0)
	c = np.mean(v0[g]/v1[g])
	return v0/c - v1

r_t = np.asarray([0.1, 0.8, 0.3])


ff_factor = 1.0
fb_factor = 0.0
num_iters = 1
h = 1.0


def run(params):
	W0, W1 = slice_params(params)

	r0 = np.ones((net_size,))
	
	r0_init = r0.copy()

	r0_h = list()
	r0p_h = list()
	e0_h, e1_h = list(), list()

	# r0_h.append(r0)	
	
	for iter_num in xrange(num_iters):
		r0p_h.append(r0)

		e0 = x - np.dot(r0, W0.T)
		e1 = r_t - np.dot(r0, W1)

		dY = ff_factor * np.dot(e0, W0) #+ fb_factor * np.dot(e1, W1.T)

		r0 = r0 + h*dY
		
		r0_h.append(r0)
		e0_h.append(e0)
		e1_h.append(e1)

	return r0_init, r0_h, r0p_h, e0_h, e1_h, r0

def runobj(params):
	W0, W1 = slice_params(params)

	r0_init, r0_h, r0p_h, e0_h, e1_h, r0 = run(params)

	return np.sum(np.square(r_t - np.dot(r0, W1)))


params = np.random.random((input_size *net_size + net_size * output_size))
drun = grad(runobj)

for i in xrange(1):
	
	r0_init, r0_h, r0p_h, e0_h, e1_h, _ = run(params)

	dparams = drun(params)
	
	W0, W1 = slice_params(params)
	dW0, dW1 = slice_params(dparams)

	dW0_h, dW1_h = list(), list()
	for r0, r0p, e0, e1 in zip(r0_h, r0p_h, e0_h, e1_h):
		
		e0 = x - np.dot(r0p, W0.T)
		# e0 = np.dot(e0, W0) + np.dot(x, W0)
		# - np.dot(np.dot(W0, r0p), W0)

		e1 = r_t - np.dot(r0, W1)

		dW1_hand = - np.outer(r0, e1)
		dW0_hand = - np.outer(e0, np.dot(e1, W1.T))

		dW0_h.append(dW0_hand)
		dW1_h.append(dW1_hand)
		

	dW0_h, dW1_h = np.asarray(dW0_h), np.asarray(dW1_h)

	shm(h*2.0*np.sum(dW0_h,0) - dW0)

	vv = np.mean(h*2.0*np.sum(dW0_h,0)- dW0, 1)

	# shl(2.0*dW1_hand[:,0], dW1[:,0])


	# params += - 0.5 * dparams 	
	if i % 10 == 0:
		print i, np.sum(np.square(r_t - np.dot(r0, W1)))





