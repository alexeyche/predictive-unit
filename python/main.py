
import time
import os
import shutil
from util import *
from dataset import ToyDataset, MNISTDataset, XorDataset, TaskType


import tensorflow.contrib.rnn as rnn
import tensorflow as tf
import numpy as np
from config import Config, dictionarize
from sklearn.metrics import log_loss
from model import *



######################################

class Optimizer(object):
    SGD = "sgd"
    ADAM = "adam"


def to_sparse_ts(d, num_iters, t=10):
    d_ts = np.zeros((num_iters,) + d.shape)
    d_ts[t] = d.copy()
    return smooth_batch_matrix(d_ts)


c = Config()
c.weight_init_factor = 1.0

c.step = 0.1
c.tau = 1.0
c.num_iters = 100

c.predictive = True
c.adaptive = False
c.adapt_gain = 10.0
c.tau_m = 1000.0

c.grad_accum_rate = 1.0/c.num_iters
c.lrate = 50.0
c.state_size = (30, )
c.lrate_factor = (1.0, 1.0)
c.fb_factor = 1.0
c.regularization = 0.0
c.optimizer = Optimizer.SGD
# c.optimizer = Optimizer.ADAM
c.epochs = 1

# ds = MNISTDataset()
ds = XorDataset()

tf.set_random_seed(13)






# def run_experiment(c, ds):
(_, input_size), (_, output_size) = ds.train_shape

# output_act = tf.nn.sigmoid if ds.task_type == TaskType.REGRESSION else tf.nn.softmax
output_act = tf.identity


xt = tf.placeholder(tf.float32, shape=(c.num_iters, None, input_size), name="xt")
yt = tf.placeholder(tf.float32, shape=(c.num_iters, None, output_size), name="yt")

is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
fb_factor = tf.placeholder(tf.float32, shape=(), name="fb_factor")

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
            output_act,
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

xt_u = tf.unstack(xt)
yt_u = tf.unstack(yt)

states_it = states
for i in xrange(c.num_iters):    
    new_states = []    

    for li, (cell, state) in enumerate(zip(net.cells, states_it)):
        with tf.variable_scope("layer{}".format(li), reuse=i>0):
            last_layer = li == len(states_it)-1
            
            input_to_layer = xt_u[i] if li == 0 else new_states[-1].a

            if last_layer:
                if isinstance(cell, OutputUnit):
                    feedback_to_layer = yt_u[i]
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

for li, s in enumerate(new_states):
    tf.summary.image("a_{}".format(li), tf.expand_dims(tf.expand_dims(s.a, 0), 3))
    # tf.summary.histogram("hist_u_{}".format(li), s.u)
    # tf.summary.histogram("hist_a_{}".format(li), s.a)
    tf.summary.scalar("error_norm_{}".format(li), tf.linalg.norm(s.e))
    tf.summary.scalar("weight_norm_{}".format(li), tf.linalg.norm(net.cells[li].F))
    tf.summary.scalar("derivative_norm_{}".format(li), tf.linalg.norm(s.dF))

if c.optimizer == Optimizer.SGD:
    optimizer = tf.train.GradientDescentOptimizer(c.lrate)
elif c.optimizer == Optimizer.ADAM:
    optimizer = tf.train.AdamOptimizer(c.lrate)
else:
    raise Exception(c.optimizer)

grads_and_vars = (
    tuple(
        (-tf.reduce_mean(s.dF, 0) * c.lrate_factor[li], l.F)
        for li, (l, s) in enumerate(zip(net.cells, new_states))
    ) 
)


apply_grads_step = tf.group(
    optimizer.apply_gradients(grads_and_vars),
    tf.assign(net.cells[0].F, tf.nn.l2_normalize(net.cells[0].F, 0))
)

# apply_grads_step = optimizer.minimize(tf.nn.l2_loss(new_states[-1].a - y)) #, var_list=[net.cells[0].F])


error_rate = (
    tf.reduce_mean(tf.cast(
        tf.not_equal(
            tf.argmax(new_outputs[-1].reconstruction[-1], axis=1),
            tf.cast(tf.argmax(y, axis=1), tf.int64)
        ), tf.float32))

    if ds.task_type == TaskType.CLASSIFICATION else
    tf.reduce_mean(tf.square(yt[10] - new_outputs[-1].reconstruction[10]))
)


tf.summary.scalar("error_rate", error_rate)


merged = tf.summary.merge_all()

#####################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

try:
    shutil.rmtree('{}/tmp/pu'.format(os.environ["HOME"]))
except:
    pass

train_writer = tf.summary.FileWriter('{}/tmp/pu/train'.format(os.environ["HOME"]), sess.graph)
test_writer = tf.summary.FileWriter('{}/tmp/pu/test'.format(os.environ["HOME"]))

init_state_fn = lambda batch_size: tuple(
    tuple(
        np.zeros([batch_size, ] + s.get_shape().as_list()[1:])
         for s in init_state
    )
    for init_state in states
)




def run(xt_v, yt_v, s_v, fb_factor_v, learn=True):
    sess_out = sess.run(
        (
            new_states,
            new_outputs,
            error_rate,
            merged,
            grads_and_vars,
        ) + ( 
            (apply_grads_step, ) if learn else tuple()
        ),
        {
            xt: xt_v,
            yt: yt_v,
            states: s_v,
            fb_factor: fb_factor_v,
            is_training: learn
        }
    )

    return (
        tuple([reset_state_fn(s) for s in sess_out[0]]), 
        sess_out[1], 
        sess_out[2],
        sess_out[3],
        sess_out[4:]
    )

outs = [PredictiveUnit.Output([],[],[],[],[]) for _ in xrange(len(net.cells))]
outs_t = [PredictiveUnit.Output([],[],[],[],[]) for _ in xrange(len(net.cells))]

perf = np.zeros(c.epochs)
fb_norm = np.zeros((c.epochs, len(net.cells)-1))
ter = np.zeros(c.epochs)




for e in xrange(c.epochs):
    states_v = [init_state_fn(ds.train_batch_size) for bi in xrange(ds.train_batches_num)]
    states_t_v = [init_state_fn(ds.test_batch_size) for bi in xrange(ds.test_batches_num)]

    train_error_rate = 0.0
    for bi in xrange(ds.train_batches_num):
        x_v, y_v = ds.next_train_batch()

        xt_v = to_sparse_ts(x_v, c.num_iters)
        yt_v = to_sparse_ts(y_v, c.num_iters)

        states_v[bi], train_outs, train_error_rate_b, summ_b, train_debug_vals = run(xt_v, yt_v, states_v[bi], c.fb_factor)
        train_writer.add_summary(summ_b, e)
        train_error_rate += train_error_rate_b/ds.train_batches_num
        
        # print e, bi, train_error_rate

    ll, test_error_rate, fb_norm_e = 0.0, 0.0, np.zeros(len(net.cells)-1)
    per_layer_error = np.zeros(len(states_t_v[0])-1)

    for bi in xrange(ds.test_batches_num):
        xtest_v, ytest_v = ds.next_test_batch()

        xtest_t_v = to_sparse_ts(xtest_v, c.num_iters, t=10)
        ytest_t_v = to_sparse_ts(ytest_v, c.num_iters, t=10)

        states_t_v[bi], test_outs, test_error_rate_b, summ_b, test_debug_vals = run(xtest_t_v, ytest_t_v, states_t_v[bi], 0.0, learn=False)
        
        test_writer.add_summary(summ_b, e)

        # ll += log_loss(yt_v, states_t_v[bi][-1].a)/ds.test_batches_num
        ll += np.sum((ytest_v - test_outs[-1][-1][10])**2.0)/ds.test_batches_num
        test_error_rate += test_error_rate_b/ds.test_batches_num
        fb_norm_e += np.asarray([np.linalg.norm(le.e) for le in states_t_v[bi][1:]])/ds.test_batches_num
        per_layer_error += np.asarray([np.sum(s.e ** 2.0) for s in states_t_v[bi][:-1] ])/ds.test_batches_num

        # print e, bi, test_error_rate

    if e > c.epochs-100:
        for li, o_v in enumerate(train_outs):
            for ot, tt in zip(outs[li], o_v):
                ot += tt
        for li, o_v in enumerate(test_outs):
            for ot, tt in zip(outs_t[li], o_v):
                ot += tt
    

    fb_norm[e] = fb_norm_e
    perf[e] = ll
    ter[e] = test_error_rate

    if e % 10 == 0:        
        # print np.linalg.norm(sess.run(net.cells[1].F))
        print "e {}, error rate {:.4f} {:.4f}, |fb| {}, error {}".format(
            e, 
            train_error_rate, test_error_rate, 
            ", ".join(["{:.4f}".format(fbe) for fbe in fb_norm[e]]),
            ", ".join(
                ["{:.4f}".format(ple) for ple in per_layer_error] + 
                ["{:.4f}".format(ll)]
                
            )
        )
    

    # return (perf, fb_norm, ter), (outs, outs_t)





# shl(np.asarray(outs[-1].a), show=False, title="Train")
# shl(np.asarray(outs_t[-1].u), show=False, title="Test")

# plt.show()

# shl(np.asarray(outs[-2].a)[:,-1], show=False, title="Train first layer")
# shl(np.asarray(outs_t[-2].a)[:,-1], show=True, title="Test first layer")


# stats, debdata = run_experiment(c, ds)

# outs, outs_t = debdata


# shl(np.tile(y_v, num_iters).reshape(num_iters, output_size), outs[1].reconstruction)
# shm(np.asarray(outs[0].a)[-1,0:200,:], np.asarray(outs_t[0].a)[-1:,0:200,:], show=False)
# shm(np.asarray(outs[1].a)[-1,0:200,:], np.asarray(outs_t[1].a)[-1:,0:200,:])

# shl(np.asarray(outs[0].a_m)[:,0,:], show=False)
# shl(np.asarray(outs[0].a)[:,0,:])