
import numpy as np
from poc.common import *
from util import *


def fuzzy_or(x, y):
	return 1.0 - (1.0 - x) * (1.0 - y)

def fuzzy_and(x, y):
	return x * y

def fuzzy_not(x):
	return 1.0 - x

def fuzzy_xor(x, y):
	return fuzzy_and(fuzzy_or(x, y), fuzzy_not(fuzzy_and(x, y)))



act = Relu()

np.random.seed(10)

input_size = 2
hidden_size = 10
output_size = 1
batch_size = 4

lrate = 0.05
Tsize = 50

dt = 0.001
tau = 2.0


W0 = np.random.randn(input_size, hidden_size)
W1 = np.random.randn(hidden_size, output_size)



xt = np.zeros((Tsize, batch_size, input_size))
yt = np.zeros((Tsize, batch_size, output_size))
y_hat_t = np.zeros((Tsize, batch_size, output_size))
y_hat_tt = np.zeros((Tsize, batch_size, output_size))


ut = np.zeros((Tsize, batch_size, hidden_size))
at = np.zeros((Tsize, batch_size, hidden_size))

utt = np.zeros((Tsize, batch_size, hidden_size))
att = np.zeros((Tsize, batch_size, hidden_size))




fb_factor = 1.0

xt[:, 0, 0], xt[:, 0, 1] = 0.0, 0.0
xt[:, 1, 0], xt[:, 1, 1] = 1.0, 1.0
xt[:, 2, 0], xt[:, 2, 1] = 0.0, 1.0
xt[:, 3, 0], xt[:, 3, 1] = 1.0, 0.0


for bi in xrange(batch_size):
	yt[:, bi] = np.expand_dims(fuzzy_xor(xt[:, bi, 0], xt[:, bi, 1]), 1)



for epoch in xrange(50000):
	u = np.zeros((batch_size, hidden_size))
	et = np.zeros((Tsize, batch_size))
	ett = np.zeros((Tsize, batch_size))


	dW1 = np.zeros(W1.shape)
	dW0 = np.zeros(W0.shape)

	de, a = 0.0, 0.0
	
	for ti in xrange(Tsize):
		x = xt[ti]
		
		u += dt * ((np.dot(x, W0) - u) + fb_factor * (np.dot(de, W1.T) - u))/tau
		
		a = act(u)

		y_hat = np.dot(a, W1)

		de = yt[ti] - y_hat

		e = np.sum(np.square(de), 1)

		if ti > Tsize/2:
			dW1 += np.dot(a.T, de)


			# dW0 += np.dot(x.T, np.dot(de, W1.T) * act.deriv(a))
			dW0 += np.dot((x - np.dot(a, W0.T)).T, a)

		et[ti] = e
		y_hat_t[ti] = y_hat.copy()
		ut[ti] = u.copy()
		at[ti] = a.copy()


	# u = np.zeros((batch_size, hidden_size))
	
	# for ti in xrange(Tsize):
	# 	x = xt[ti]
		
	# 	u += dt * (np.dot(x, W0) - u)/tau
		
	# 	a = act(u)

	# 	y_hat = np.dot(a, W1)

	# 	de = yt[ti] - y_hat

	# 	e = np.sum(np.square(de), 1)

	# 	ett[ti] = e
	# 	y_hat_tt[ti] = y_hat.copy()
	# 	utt[ti] = u.copy()
	# 	att[ti] = a.copy()


	W0 += dW0 * lrate
	W1 += dW1 * lrate
	if epoch % 100 == 0:
		print "Epoch {}, error: {}".format(epoch, np.sum(et[:-(Tsize/2)]))
