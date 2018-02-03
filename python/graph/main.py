#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from sklearn.datasets import make_classification
from util import *
from scipy import sparse as nps


def stv_to_dense(m):
	md = np.zeros(m.dense_shape)
	md[m.indices[:,0], m.indices[:,1]] = m.values
	return md

def from_coo_to_dense(vals, idx, shape):

	return nps.coo_matrix((vals, ([t[0] for t in idx], [t[1] for t in idx])), shape=shape).todense()

def top_k_sparse_tensor(x, k):
	x_k, x_k_idx = tf.nn.top_k(x, k=k)


	return tf.sparse_reorder(tf.SparseTensor(
		values=tf.reshape(x_k, [-1]), 
		indices=tf.cast(
			tf.transpose(tf.stack([
				tf.reshape(tf.tile(tf.expand_dims(tf.range(0, tf.shape(x)[0]), 1), [1, k]), [-1]), 
				tf.reshape(x_k_idx, [-1])
			])),
			tf.int64
		), 
		dense_shape=tf.shape(u, out_type=tf.int64)
	))


batch_size = 200

input_size = 20
layer_size = 1000

n_classes = 2
seed = 1
k = int(0.05*layer_size)


np.random.seed(seed)
tf.set_random_seed(seed)


x_values, y_values = make_classification(
    n_samples=batch_size,
    n_features=input_size, 
    n_informative=input_size/2, 
    n_redundant=1, 
    n_repeated=1,
    n_clusters_per_class=2,
    n_classes=n_classes,
    random_state=seed
)




x = tf.placeholder(tf.float32, shape=(None, input_size), name="x")



Widx = tf.placeholder(shape=(None, None), dtype=tf.int64)
Wval = tf.placeholder(shape=(None), dtype=tf.float32)

Ridx = tf.placeholder(shape=(None, None), dtype=tf.int64)
Rval = tf.placeholder(shape=(None), dtype=tf.float32)


W = tf.SparseTensor(indices=Widx, values=Wval, dense_shape=[input_size, layer_size])
R = tf.SparseTensor(indices=Ridx, values=Rval, dense_shape=[layer_size, layer_size])


u = tf.transpose(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(W), tf.transpose(x)))

u_new = tf.sparse_tensor_to_dense(top_k_sparse_tensor(u, k))

u_new_rec = tf.transpose(tf.sparse_tensor_dense_matmul(R, tf.transpose(u_new)))




sess = tf.Session()
sess.run(tf.global_variables_initializer())


Ws = 1.0


Widx_v, Wval_v, p = [], [], 0.01
for i in xrange(input_size):
	for j in xrange(layer_size):
		if np.random.random() < p:
			Widx_v.append((i, j))
			Wval_v.append(Ws)

Ridx_v, Rval_v, p = [], [], 0.001
for i in xrange(layer_size):
	for j in xrange(layer_size):
		if np.random.random() < p:
			Ridx_v.append((i, j))
			Rval_v.append(Ws)





W_v = from_coo_to_dense(Wval_v, Widx_v, (input_size, layer_size))
R_v = from_coo_to_dense(Rval_v, Ridx_v, (layer_size, layer_size))


u_v, u_new_v, u_new_rec_v = sess.run(
	(u, u_new, u_new_rec), 
	{
		x: x_values,
		Widx: Widx_v,
		Wval: Wval_v,
		Ridx: Ridx_v,
		Rval: Rval_v,
	}
)


# u_new_v = stv_to_dense(u_new_v)

