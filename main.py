
import time
from util import *
from dataset import get_toy_data_baseline, one_hot_encode


import tensorflow.contrib.rnn as rnn
import tensorflow as tf
import numpy as np
from config import Config, dictionarize
from sklearn.metrics import log_loss
from model import *
Dataset = tf.data.Dataset

tf.set_random_seed(3)
np.random.seed(3)


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

class Optimizer(object):
    SGD = "sgd"
    ADAM = "adam"



def run_experiment(c):
    x = tf.placeholder(tf.float32, shape=(None, input_size), name="x")
    y = tf.placeholder(tf.float32, shape=(None, output_size), name="y")
    is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
    fb_factor = tf.placeholder(tf.float32, shape=(), name="fb_factor")

    print [
        c.state_size[li+1] if li < len(c.state_size) - 1 else output_size
        for li, ss in enumerate(c.state_size)
    ]
    net = FeedbackNet(*[
            PredictiveUnit(
                c.state_size[li-1] if li > 0 else input_size,
                ss, 
                c.state_size[li+1] if li < len(c.state_size) - 1 else output_size,
                c, 
                tf.nn.relu, 
                fb_factor if li == len(c.state_size) - 1 else 1.0, 
                is_training
            )
            for li, ss in enumerate(c.state_size)
        ] + [
            OutputUnit(
                c.state_size[-1], 
                output_size, 
                output_size, 
                c, 
                tf.nn.softmax, 
                0.0, 
                is_training
            )
        ]
    )


    states = tuple(
        PredictiveUnit.State(*tuple(
            tf.placeholder(tf.float32, (None,) + (size if isinstance(size, tuple) else (size,)))
            for size in cell.state_size
        ))
        for cell in net.cells
    )

    Bs = [tf.Variable(tf.random_normal([output_size, cell.layer_size])) for cell in net.cells[:-1]]

    new_outputs = [PredictiveUnit.Output([],[],[],[],[]) for _ in xrange(len(net.cells))]

    states_it = states
    for i in xrange(c.num_iters):    
        new_states = []    

        for li, (cell, state) in enumerate(zip(net.cells, states_it)):
            with tf.variable_scope("layer{}".format(li), reuse=i>0):
                last_layer = li == len(states_it)-1
                
                input_to_layer = x if li == 0 else new_states[-1].a

                if last_layer:
                    if isinstance(cell, OutputUnit):
                        feedback_to_layer = y
                    elif isinstance(cell, PredictiveUnit):
                        feedback_to_layer = tf.zeros((tf.shape(x)[0], cell.feedback_size))
                else:
                    feedback_to_layer = states_it[li+1].e
                    # feedback_to_layer = tf.matmul(states_it[-1].e, Bs[li])
                
                o, s = cell((input_to_layer, feedback_to_layer), state)

                new_states.append(s)
                
                for v, dst_list in zip(o, new_outputs[li]):
                    dst_list.append(v)

        states_it = tuple(new_states)



    if c.optimizer == Optimizer.SGD:
        optimizer = tf.train.GradientDescentOptimizer(c.lrate)
    elif c.optimizer == Optimizer.ADAM:
        optimizer = tf.train.AdamOptimizer(c.lrate)
    else:
        raise Exception(c.optimizer)

    grads_and_vars = tuple(
        (-tf.reduce_mean(s.dF, 0) * (c.net_lrate_factor if li < len(net.cells)-1 else c.out_lrate_factor), l.F) 
        for li, (l, s) in enumerate(zip(net.cells, new_states))
    )

    # grads_and_vars = ((-tf.reduce_mean(new_states[-1].dF,0), net.cells[-1].F),)

    apply_grads_step = tf.group(
        optimizer.apply_gradients(grads_and_vars),
    )

    # apply_grads_step = optimizer.minimize(tf.nn.l2_loss(new_states[-1].a - y), var_list=[net.cells[0].F])


    error_rate = tf.reduce_mean(tf.cast(
        tf.not_equal(
            tf.argmax(new_outputs[-1].reconstruction[-1], axis=1),
            tf.cast(tf.argmax(y, axis=1), tf.int64)
        ), tf.float32))



    #####################################

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    init_state_fn = lambda batch_size: tuple(
        tuple(
            np.zeros([batch_size, ] + s.get_shape().as_list()[1:])
             for s in init_state
        )
        for init_state in states
    )



    
    
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
                fb_factor: fb_factor_v,
                is_training: learn
            }
        )

        return (
            tuple([reset_state_fn(s) for s in sess_out[0]]), 
            sess_out[1], 
            sess_out[2],
        )

    outs = [PredictiveUnit.Output([],[],[],[],[]) for _ in xrange(len(net.cells))]
    outs_t = [PredictiveUnit.Output([],[],[],[],[]) for _ in xrange(len(net.cells))]

    perf = np.zeros(c.epochs)
    fb_norm = np.zeros(c.epochs)
    ter = np.zeros(c.epochs)

    states_v = init_state_fn(x_v.shape[0])
    states_t_v = init_state_fn(xt_v.shape[0])

    for e in xrange(c.epochs):

        # if e > 500:
        #     fb_factor_v = 1.0

        states_v, train_outs, train_error_rate = run(x_v, y_v, states_v, c.fb_factor)
        states_t_v, test_outs, test_error_rate = run(xt_v, yt_v, states_t_v, 0.0, learn=False)
        
        ll = log_loss(yt_v, states_t_v[-1].a)

        if e > c.epochs-20:
            for li, o_v in enumerate(train_outs):
                for ot, tt in zip(outs[li], o_v):
                    ot += tt
            for li, o_v in enumerate(test_outs):
                for ot, tt in zip(outs_t[li], o_v):
                    ot += tt
        

        fb_norm[e] = np.linalg.norm(states_t_v[-1].e)
        perf[e] = ll
        ter[e] = test_error_rate 

        if e % 100 == 0:        
            # print np.linalg.norm(sess.run(net.cells[1].F))
            print "e {}, error rate {:.4f}, |fb| {:.4f}, error {}".format(
                e, 
                test_error_rate,
                fb_norm[e],
                ", ".join(
                    ["{:.4f}".format(np.sum(s.e ** 2.0)) for s in states_t_v[:-1] ] + 
                    # ["{:.4f}".format(np.sum((states_t_v[-1].a - yt_v) ** 2.0))]
                    ["{:.4f}".format(ll)]
                    
                )
            )
        

    return (perf, fb_norm, ter), (outs, outs_t)





c = Config()
c.weight_init_factor = 1.0
c.state_size = (50,)

c.step = 0.001
c.tau = 10.0
c.num_iters = 10

c.adaptive = True
c.adapt_gain = 10.0
c.tau_m = 1000.0

c.grad_accum_rate = 1.0/c.num_iters
c.lrate = 1.0 * 10.0
c.net_lrate_factor = 1.0
c.out_lrate_factor = 0.0
c.optimizer = Optimizer.SGD
c.fb_factor = 1.0
# c.optimizer = Optimizer.ADAM
c.epochs = 5000

stats, debdata = run_experiment(c)


# shl(np.tile(y_v, num_iters).reshape(num_iters, output_size), outs[1].reconstruction)
# shm(np.asarray(outs[0].a)[-1,0:200,:], np.asarray(outs_t[0].a)[-1:,0:200,:], show=False)
# shm(np.asarray(outs[1].a)[-1,0:200,:], np.asarray(outs_t[1].a)[-1:,0:200,:])

# shl(np.asarray(outs[0].a_m)[:,0,:], show=False)
# shl(np.asarray(outs[0].a)[:,0,:])