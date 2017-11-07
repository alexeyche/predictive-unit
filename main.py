
import time
from util import *
from dataset import get_toy_data_baseline, one_hot_encode


import tensorflow.contrib.rnn as rnn
import tensorflow as tf
import numpy as np
from config import Config

from model import *

tf.set_random_seed(1)
np.random.seed(1)


x_v, target_v = get_toy_data_baseline()
input_size = x_v.shape[1]

test_prop = x_v.shape[0]/5

xt_v = x_v[:test_prop]
target_t_v = target_v[:test_prop]

x_v = x_v[test_prop:]
target_v = target_v[test_prop:]

y_v = one_hot_encode(target_v)
yt_v = one_hot_encode(target_t_v)


output_size = y_v.shape[1]



######################################


state_size = 50


x = tf.placeholder(tf.float32, shape=(None, input_size), name="x")
y = tf.placeholder(tf.float32, shape=(None, output_size), name="y")


c = Config()
c.weight_init_factor = 1.0
c.step = 0.01
c.tau = 10.0
num_iters = 30
c.grad_accum_rate = 1.0/num_iters
c.fb_factor = tf.placeholder(tf.float32, shape=(), name="fb_factor")
lrate = 0.005


net = FeedbackNet(
    PredictiveUnit(input_size, state_size, output_size, c, tf.nn.relu),
    OutputUnit(state_size, output_size, output_size, c, tf.identity)
)


states = tuple(
    PredictiveUnit.State(*tuple(
        tf.placeholder(tf.float32, (None,) + (size if isinstance(size, tuple) else (size,)))
        for size in cell.state_size
    ))
    for cell in net.cells
)



new_outputs = [PredictiveUnit.Output([],[],[],[]) for _ in xrange(len(net.cells))]

states_it = states
for i in xrange(num_iters):
    new_states = []    

    for li, (cell, state) in enumerate(zip(net.cells, states_it)):
        input_to_layer = x if li == 0 else new_states[-1].a
        feedback_to_layer = y if li == len(states_it)-1 else states_it[li+1].e
        
        o, s = cell((input_to_layer, feedback_to_layer), state)

        new_states.append(s)
        
        for v, dst_list in zip(o, new_outputs[li]):
            dst_list.append(v)

    states_it = tuple(new_states)


# optimizer = tf.train.GradientDescentOptimizer(lrate)
optimizer = tf.train.AdamOptimizer(lrate)

grads_and_vars = tuple(
    (-tf.reduce_mean(s.dF, 0), l.F) 
    for l, s in zip(net.cells, new_states)
)

apply_grads_step = tf.group(
    optimizer.apply_gradients(grads_and_vars),
)

error_rate = tf.reduce_mean(tf.cast(
    tf.not_equal(
        tf.argmax(new_outputs[-1].reconstruction[-1], axis=1),
        tf.cast(tf.argmax(y, axis=1), tf.int64)
    ), tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())


init_state_fn = lambda batch_size: tuple(
    tuple(
        np.zeros([batch_size, ] + s.get_shape().as_list()[1:])
         for s in init_state
    )
    for init_state in states
)





outs = [PredictiveUnit.Output([],[],[],[]) for _ in xrange(len(net.cells))]
    
    
def run(x_v, y_v, s_v, fb_factor_v, learn=True):
    sess_out = sess.run(
        (
            new_states,
            new_outputs,
            error_rate,
        ) + ( 
            (apply_grads_step, ) if learn else tuple()
        ),
        {
            x: x_v,
            y: y_v,
            states: s_v,
            c.fb_factor: fb_factor_v
        }
    )

    return (
        tuple([reset_state_fn(s) for s in sess_out[0]]), 
        sess_out[1], 
        sess_out[2],
    )


for e in xrange(200):
    states_v = init_state_fn(x_v.shape[0])
    states_t_v = init_state_fn(xt_v.shape[0])
    
    
    states_v, train_outs, train_error_rate = run(x_v, y_v, states_v, 0.01)
    states_t_v, test_outs, test_error_rate = run(xt_v, yt_v, states_t_v, 0.0, learn=False)
    
    for li, o_v in enumerate(train_outs):
        for ot, tt in zip(outs[li], o_v):
            ot += tt
    
    print "e {}, train error {:.4f}, test error {:.4f},  error {}".format(
        e, 
        train_error_rate,
        test_error_rate,
        ", ".join(
            ["{:.4f}".format(np.sum(s.e ** 2.0)) for s in states_v[:-1] ] + 
            ["{:.4f}".format(np.sum((states_v[-1].a - y_v) ** 2.0))]
        )
    )


# shl(np.tile(y_v, num_iters).reshape(num_iters, output_size), outs[1].reconstruction)

