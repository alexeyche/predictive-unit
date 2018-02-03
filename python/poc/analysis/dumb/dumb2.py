from util import *
import tensorflow as tf



n_in, n_out = 3, 5


np.random.seed(11)

W = tf.placeholder(shape=(n_in, n_out), dtype=tf.float32)

y = tf.placeholder(shape=(n_out,), dtype=tf.float32)

part = tf.matmul(tf.transpose(W), W)
o = tf.reduce_sum(tf.matmul(tf.expand_dims(y, 0), part))


## y * o
## y' * o + y * o'

df = tf.gradients(part, [W])[0]
df2 = tf.gradients(o, [W])[0]
df_part = tf.gradients(o, [part])[0]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

y_v = np.random.random((n_out,))
# y_v[0] = 0.5

W_v = np.random.random((n_in, n_out))

df_v, df2_v, df_part_v, W_v, y_v, part_v, o_v = sess.run((df, df2, df_part, W, y, part, o), {y: y_v, W: W_v})


df_h = np.asarray([np.sum(W_v, 1)*2.0]*5).T
df_part_h = np.asarray([y_v]*5).T


assert np.sum(np.square(df_h - df_v)) < 1e-10
assert np.sum(np.square(df_part_h - df_part_v)) < 1e-10


# df_h = np.asarray([np.sum(W_v, 1)]*5).T + np.asarray([np.sum(W_v, 0)]*3)

print np.dot(df_part_h, W_v.T) + df_h.T - df2_v.T
# np.dot(df_part_h, W_v.T) + df_h.T + np.dot(np.asarray([y_v]*5), W_v.T) - df2_v.T

print np.dot(df_part_h.T, W_v.T) + np.dot(df_part_h, W_v.T) - df2_v.T
print np.dot(df_part_h.T + df_part_h, W_v.T) - df2_v.T

###

dd = np.zeros(W_v.shape)

for i in xrange(n_in):
	for j in xrange(n_out):
		delta_val = 1e-03
		delta = np.zeros(W_v.shape)
		delta[i,j] = delta_val

		o_l = sess.run(o, {y: y_v, W: W_v+delta})
		o_r = sess.run(o, {y: y_v, W: W_v-delta})

		dd[i,j] = (o_l - o_r)/(2.0*delta_val)




