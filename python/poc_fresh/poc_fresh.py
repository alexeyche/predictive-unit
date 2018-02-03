
import numpy as np
from util import *
from poc.common import *
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss

# x = np.asarray([
#     [0.0, 1.0, 0.0, 1.0],
#     [1.0, 0.0, 1.0, 0.0],
# ])

# y_t = np.asarray([
# 	[0.0],
# 	[1.0]
# ])

act_o = Sigmoid()
act = Relu()


x = np.asarray([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

y_t = np.asarray([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])



# np.random.seed(19)


batch_size = x.shape[0]
input_size, output_size = x.shape[1], y_t.shape[1]
net_size = 25

weight_factor = 0.01

W0 = weight_factor/2.0 - weight_factor*np.random.random((input_size, net_size))
W1 = weight_factor/2.0 - weight_factor*np.random.random((net_size, output_size))
b0 = np.zeros((net_size,))
b1 = np.zeros((output_size,))

def simple_dynamics(u0, a0, x, x_fb, fb_factor):
	du = np.dot(x, W0) + b0 + np.dot(x_fb, W1.T) * fb_factor - u0
	return du, x

def predictive_dynamics(u0, a0, x, x_fb, fb_factor):
	ff = x - np.dot(a0, W0.T)
	fb = np.dot(x_fb, W1.T)
	du = np.dot(ff, W0) + b0 + fb * fb_factor - u0
	return du, ff


dudt = simple_dynamics

def run_network(num_iters, fb_factor, clamp, u0_given=None):
	u0 = u0_given if not u0_given is None else np.zeros((batch_size, net_size))

	a0 = np.zeros((batch_size, net_size))

	y = np.zeros((batch_size, output_size))

	a0h = np.zeros((num_iters, batch_size, net_size))
	u0h = np.zeros((num_iters, batch_size, net_size))
	ff0h = np.zeros((num_iters, batch_size, input_size))
	a1h = np.zeros((num_iters, batch_size, output_size))
	deh = np.zeros((num_iters, batch_size, output_size))
	eh = np.zeros((num_iters, batch_size))

	for i in xrange(num_iters):
		du0, ff0 = dudt(u0, a0, x, y_t if clamp else y, fb_factor)
		u0 += h * du0

		a0 = act(u0)

		y = act_o(np.dot(a0, W1) + b1)

		de = y_t - y
		
		u0h[i] = u0.copy()
		a0h[i] = a0.copy()
		a1h[i] = y.copy()
		deh[i] = de.copy()
		eh[i] = np.sum(de ** 2.0, 0)
		ff0h[i] = ff0.copy()

	return (
		u0h,
		a0h,
		ff0h,
		a1h,
		deh,
		eh
	)

h = 0.1
lrate = 0.1
num_iters = 50
fb_factor = 0.1


for e in xrange(2000):
	u0_free, a0_free, ff0_free, a1_free, de_free, e_free = run_network(
		num_iters, 
		fb_factor, 
		clamp=False,
	)
	u0_clamp, a0_clamp, ff0_clamp, a1_clamp, de_clamp, e_clamp = run_network(
		num_iters, 
		fb_factor, 
		clamp=True, 
		# u0_given=u0_free[-1].copy()
	)

	dW0_c = np.dot(ff0_clamp[-1].T, a0_clamp[-1])
	dW0_f = np.dot(ff0_free[-1].T, a0_free[-1])
	
	dW0 = dW0_c - dW0_f


	dW1_c = np.dot(a0_clamp[-1].T, y_t)
	dW1_f = np.dot(a0_free[-1].T, a1_free[-1])
	
	dW1 = dW1_c - dW1_f

	db0 = np.sum(a0_clamp[-1], 0) - np.sum(a0_free[-1], 0)
	db1 = np.sum(y_t, 0) - np.sum(a1_free[-1], 0)

	W0 += lrate * dW0 
	W1 += lrate * dW1

	b0 += lrate * db0
	b1 += lrate * db1

	if e % 200 == 0:
		print e, log_loss(y_t, a1_free[-1])

# shl(u0_free[:,0], show=False, title="First u0 free")
# shl(u0_free[:,1], show=False, title="Second u0 free")

# shl(u0_clamp[:,0], show=False, title="First u0 clamp")
# shl(u0_clamp[:,1], show=False, title="Second u0 clamp")

shl(a1_free, show=False, title="Y free")
shl(a1_clamp, show=False, title="Y clamp")

plt.show()


