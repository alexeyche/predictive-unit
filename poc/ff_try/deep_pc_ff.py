
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *

np.random.seed(5)

act = Relu()
act_o = Linear()

input_size = 100

x = np.random.randn(input_size)

lrule = Learning.BP


y_t = np.asarray([0.8, 0.2])

net_structure = (50, 20, 2)


S = lambda x: np.log(1.0 + np.square(x))
dS = lambda x: 2.0 * x / (np.square(x) + 1.0)


W = list(
    np.random.random((net_structure[li-1] if li > 0 else input_size, size))*0.1
    for li, size in enumerate(net_structure)
)


Wcp = [w.copy() for w in W]

B = list(
    np.random.random((net_structure[-1], size))*1.0
    for li, size in enumerate(net_structure[:-1])
)


fb_factor = 1.0
tau = 5.0
num_iters = 500

step = 0.1

tau_apical = 2.0
tau_basal = 2.0

lrate = 0.001


# sp_code = False
predictive_output = False

s_h = [np.zeros((num_iters, ns)) for ns in net_structure]
r_h = [np.zeros((num_iters, ns)) for ns in net_structure]
e_h = [np.zeros((num_iters, 2)) for _ in net_structure]
y_h = np.zeros((num_iters))


s = [np.zeros(ns) for ns in net_structure]
e = [np.zeros((2, ns)) for ns in net_structure[:-1]]
r = [np.zeros(ns) for ns in net_structure]

for i in xrange(num_iters):    
    for li in xrange(len(net_structure)-1):
        input_to_layer = x if li == 0 else r[li-1]
        
        d0 = np.dot(input_to_layer, W[li]) - s[li]
        e[li][0] = d0
        d1 = np.dot(r[li+1], W[li+1].T) - s[li]
        e[li][1] = d1
        
        s[li] += step * (d0 + d1)/tau

        r[li] = act(s[li])
        
    
    s[-1] = np.dot(r[-2], W[-1])
    # r[-1] = act_o(s[-1])
    # e[-1][0] = np.dot(r[-1] - y_t, W[-1].T)
    r[-1] = y_t - act_o(s[-1])

    # for li in xrange(len(net_structure)-1):
    #     e[li+1] = np.dot(- (y_t - r[-1]), B[li])

    # print tuple(ee.shape for ee in e)
    # raise Exception
    error = tuple(np.linalg.norm(ee, axis=1) for ee in e)

#     # dW0 = 
#     dW0 = -np.outer(in0, r0 * act.deriv(top_down0))

     # dW = tuple(-np.outer(ee, np.dot(e, W[1].T) ) for ee, li in enumerate(e))
    dW = [np.zeros(w.shape) for w in W]
    for li, rr in enumerate(r[:-1]):
        input_to_layer = x if li == 0 else r[li-1]

        dW[li] += np.outer(-input_to_layer, rr)
        dW[li+1] += np.outer(r[li+1], rr).T

    dW[-1] = np.outer(r[-2], y_t - r[-1])
    
      # dW0 = -np.outer(in0, r0 * act.deriv(top_down0))
#     dW1 = -np.outer(r0, e)
    for li in xrange(len(net_structure)):
        W[li] += lrate * dW[li]

    

    # for li, dWl in enumerate(dW):
    #     W[li] += lrate*dWl

#     W[0] -= lrate * dW0
#     W[1] -= lrate * dW1

    print "i {}, error {}".format(i, ", ".join(["{:.4f} {:.4f}".format(*ee) for ee in error]))

    for s_hl, sl in zip(s_h, s):
        s_hl[i] = sl.copy()

    for r_hl, rl in zip(r_h, r):
        r_hl[i] = rl.copy()

    for e_hl, error_l in zip(e_h, error):
        e_hl[i] = error_l

#     h0_h[i] = h0.copy()
#     e0_h[i] = error[0]
#     h1_h[i] = h1.copy()
#     e1_h[i] = error[1]

shl(*e_h)

# shl(h_h[-1][:,0], np.asarray([y_t[0]]*num_iters),np.asarray([3.0]*num_iters), show=False)
# shl(h_h[-1][:,1], np.asarray([y_t[1]]*num_iters),np.asarray([3.0]*num_iters))
