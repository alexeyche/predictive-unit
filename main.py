
import time
from util import *
from dataset import get_toy_data, one_hot_encode


import tensorflow.contrib.rnn as rnn
import tensorflow as tf
import numpy as np
from config import Config

from model import *

# input_size = 20
# x_v, target_v = get_toy_data(input_size, 2000, seed = 5)

# test_prop = x_v.shape[0]/5

# xt_v = x_v[:test_prop]
# target_t_v = target_v[:test_prop]

# x_v = x_v[test_prop:]
# target_v = target_v[test_prop:]

# y_v = one_hot_encode(target_v)
# yt_v = one_hot_encode(target_t_v)


# output_size = y_v.shape[1]

# tf.set_random_seed(1)
# np.random.seed(1)

input_size = 2
output_size = 1
batch_size = 1

x_v = np.ones((batch_size, input_size))
y_v = np.asarray([[0.5]])


######################################


state_size = 1


x = tf.placeholder(tf.float32, shape=(None, input_size), name="x")
y = tf.placeholder(tf.float32, shape=(None, output_size), name="y")


c = Config()
c.weight_init_factor = 1.0
c.step = 0.1
c.tau = 5.0
c.grad_accum_rate = 0.001
c.fb_factor = 1.0
lrate = 1e-05


net = FeedbackNet(
    PredictiveUnit(input_size, state_size, output_size, c, tf.identity),
    OutputUnit(state_size, output_size, output_size, c, tf.identity)
)


states = tuple(
    PredictiveUnit.State(*tuple(
        tf.placeholder(tf.float32, (None,) + (size if isinstance(size, tuple) else (size,)))
        for size in cell.state_size
    ))
    for cell in net.cells
)


new_states = []
new_outputs = []

for li, (cell, state) in enumerate(zip(net.cells, states)):
    input_to_layer = x if li == 0 else states[li-1].a
    feedback_to_layer = y if li == len(states)-1 else states[li+1].e
    
    o, s = cell((input_to_layer, feedback_to_layer), state)

    new_states.append(s)
    new_outputs.append(o)




optimizer = tf.train.GradientDescentOptimizer(lrate)
grads_and_vars = tuple(
    (-tf.reduce_mean(s.dF, 0), l.F) 
    for l, s in zip(net.cells, states)
)


apply_grads_step = tf.group(
    optimizer.apply_gradients(grads_and_vars),
)

# error_rate = tf.reduce_mean(tf.cast(
#     tf.not_equal(
#         tf.argmax(debug[-1].reconstruction, axis=1),  
#         tf.cast(tf.argmax(y, axis=1), tf.int64)
#     ), tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())


batch_size_v = x_v.shape[0]

init_state_fn = lambda: tuple(
    tuple(
        np.zeros([batch_size_v, ] + s.get_shape().as_list()[1:])
         for s in init_state
    )
    for init_state in states
)

states_v = init_state_fn()

outs = [PredictiveUnit.Output([],[],[],[]) for _ in xrange(len(net.cells))]

for i in xrange(1000):
    feeds = {
        x: x_v,
        y: y_v,
        states: states_v
    }
    
    ns_v, no_v, _ = sess.run(
        (
            new_states,
            new_outputs,
            apply_grads_step
        ),
        feeds
    )

    states_v = tuple(ns_v)
    
    for li, o_v in enumerate(no_v):
        for ot, tt in zip(outs[li], o_v):
            ot.append(tt)

    print "i {},  error {}".format(i, ", ".join(["{:.4f}".format(np.linalg.norm(s.e)) for s in states_v]))


shl(outs[1].e)