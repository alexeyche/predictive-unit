
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *

# np.random.seed(5)

act = Linear()
act_o = Linear()

input_size = 2
x = np.ones((input_size,))

lrule = Learning.BP


y_t = np.asarray([0.5, 0.5])




net_structure = (3,2)


S = lambda x: np.log(1.0 + np.square(x))
dS = lambda x: 2.0 * x / (np.square(x) + 1.0)


# W = list(
#     np.random.random((net_structure[li-1] if li > 0 else input_size, size))*1.0
#     for li, size in enumerate(net_structure)
# )

W = list([
    np.asarray([
        [ 0.77172887,  0.92807782,  0.98455656], 
        [ 0.48378634,  0.64331949,  0.6040535 ]
    ]),
    np.asarray([
        [ 0.65488744,  0.46419418],
        [ 0.55719793,  0.27298439],
        [ 0.88136756,  0.64968431]
    ])
])


Wcp = [w.copy() for w in W]

B = np.random.random((net_structure[-1], net_structure[-2]))*1.0

fb_factor = 1.0
tau = 5.0
num_iters = 200
h = 0.01

tau_apical = 2.0
tau_basal = 2.0

lrate = 0.01


# sp_code = False
predictive_output = False

h0_h = np.zeros((num_iters, net_structure[0]))
e0_h = np.zeros((num_iters))
h1_h = np.zeros((num_iters, net_structure[1]))
e1_h = np.zeros((num_iters))
y_h = np.zeros((num_iters))
in0_h = np.zeros((num_iters, input_size))

h0 = np.zeros(net_structure[0])
r0 = np.zeros(net_structure[0])
h1 = np.zeros(net_structure[1])
r1 = np.zeros(net_structure[1])

e = np.zeros((net_structure[-1]))

dW = [np.zeros(w.shape) for w in W]


for i in xrange(100):
    in0 = x - np.dot(r0, W[0].T)
    top_down0 = e
    
    h0 += h * (np.dot(in0, W[0]) + fb_factor * np.dot(top_down0, W[1].T))
    
    r0 = act(h0)

    
    if predictive_output:
        # in1 = np.dot(e, W[1].T)
        # h1 += h * np.dot(in1, W[1])

        in1 = y_t - (r1 - np.dot(r0, W[1]))
        h1 += h * in1
    else:
        h1 = np.dot(r0, W[1])
    
    r1 = act(h1)

    e = y_t - r1

    error = np.asarray((
        np.sum(in0 ** 2.0),
        np.sum(e ** 2.0),
    ))


    
    # dW0 = -np.outer(in0, np.dot(e, W[1].T) )
    
    dW0 = np.outer(in0, r0)
    dW1 = np.outer(r0, e)
    
    W[0] += lrate * dW0
    W[1] += lrate * dW1

    # print h0, "|", h1
    # print np.mean(dW0, 0), np.mean(dW1, 0)
    print "i {}, error {}".format(i, ", ".join(["{:.4f}".format(ee) for ee in error]))

    h0_h[i] = h0.copy()
    e0_h[i] = error[0]
    h1_h[i] = h1.copy()
    e1_h[i] = error[1]
    in0_h[i] = in0.copy()

# shl(h1_h, np.asarray([y_t]*num_iters),np.asarray([3.0]*num_iters))
# shl(e1_h)
# shl(h1_h)

