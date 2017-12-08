

from util import *

import tensorflow as tf

np.random.seed(11)
tf.set_random_seed(11)

def batch_outer(left, right):
    return np.asarray([np.outer(left[i], right[i]) for i in xrange(left.shape[0])])


def test_g(l, r):
	assert np.all(np.square(l - r) < 1e-07), "Diff is not that small:\n{}".format(np.square(l - r))

input_size = 2
net_size = 2
output_size = 1


ff_factor = 1.0
fb_factor = 0.0
num_iters = 1
h = 1.0



x = tf.placeholder(tf.float32, shape=(None, input_size), name="x")
y = tf.placeholder(tf.float32, shape=(None, output_size), name="y")
a = tf.placeholder(tf.float32, shape=(None, net_size), name="a")
u = tf.placeholder(tf.float32, shape=(None, net_size), name="u")

W0 = tf.Variable(np.asarray([[1.0, 1.0], [1.0, 1.0]]), dtype=tf.float32)
W1 = tf.Variable(np.asarray([[1.0], [1.0]]), dtype=tf.float32)



u_w, a_w = u, a
ee, aa = [], []
for iter_num in xrange(num_iters):
	x_hat = tf.matmul(a_w, tf.transpose(W0))
	e = x - x_hat
	
	dudt = tf.matmul(e, W0)

	u_w = u_w + h * dudt

	a_w = tf.identity(u_w) # activation	

	ee.append(e)
	aa.append(a_w)

y_hat = tf.matmul(a_w, W1)

loss = tf.nn.l2_loss(y_hat - y)

u_wg, a_wg, W0g, W1g = tf.gradients(loss, [u_w, a_w, W0, W1])
dW0_0 = tf.gradients(u_w, [W0])[0]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

bs = 3

x_v = np.ones((bs, input_size))
x_v[0,0] = 5.0
y_v = np.ones((bs, output_size))
u0_v = np.zeros((bs, net_size,))
a0_v = np.zeros((bs, net_size,))
a0_v[:,0] = 0.5


u_v, a_v, y_hatv, lossv, W0v, W1v, u_gv, a_gv, W0gv, W1gv, eev, aav, dW0_0v = \
	sess.run(
		(u_w, a_w, y_hat, loss, W0, W1, u_wg, a_wg, W0g, W1g, ee, aa, dW0_0), 
		{
			x: x_v,
			y: y_v,
			u: u0_v,
			a: a0_v
		}
	)



u_gv_hand = np.dot(y_hatv - y_v, W1v.T)
W1gv_hand = np.dot(a_v.T, (y_hatv - y_v))


# a_p = np.expand_dims(a0_v, 1)

# a_p = np.tile(np.expand_dims(a0_v, 1), (1, net_size, 1))
a_p = np.expand_dims(a0_v, 1)
# x_p = np.tile(np.expand_dims(x_v, 1), (1, net_size, 1))
# u_gv_hand_p = np.tile(np.expand_dims(u_gv_hand, 1), (1, net_size, 1))


# e_0 = np.expand_dims(x_v, 2) - np.transpose(np.dot(a_p + np.transpose(a_p, (0, 2, 1)), W0v.T), (0, 2, 1))
# e_0 = np.transpose(e_0, (0, 2, 1))

e_0 = np.expand_dims(x_v, 1) - np.expand_dims(a0_v, 1) + np.transpose(np.expand_dims(a0_v, 1), (0, 2, 1))

# e_0 = x_v - np.dot(a0_v + a0_v.T, W0v.T)

# e_0 = x_v - np.dot(np.tile(a0_v, (2, 1)).T + np.tile(a0_v, (2, 1)), W0v.T)

# e_0 = x_v - (np.dot(a0_v, W0v.T) + np.dot(a0_v, W0v.T).T)

# -W1*(-2*W0*y + x)*(-W1*(-W0**2*y + W0*x) + yt)

# W0gv_hand = np.dot(e_0, u_gv_hand).T
# W0gv_hand = np.dot(e_0, u_gv_hand)

np.sum(batch_outer(e_0, u_gv_hand), 0)
W0gv_hand = np.dot(np.mean(e_0, 1).T, u_gv_hand).T
# np.asarray([np.dot(e_0[bi], np.expand_dims(u_gv_hand[bi], 2)) for bi in xrange(bs)])

# dW0 = np.tile(np.sum(x_v, 0), (net_size, 1)).T - np.sum(np.dot(a_p + np.transpose(a_p, (0, 2, 1)), W0v.T), 0).T


test_g(u_gv, u_gv_hand)
test_g(W1gv, W1gv_hand)
test_g(W0gv, W0gv_hand)







