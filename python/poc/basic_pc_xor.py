
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *

# np.random.seed(5)

act = Linear()
act_o = Sigmoid()

input_size = 2

lrule = Learning.BP

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

batch_size = x.shape[0]



net_structure = (1,1)


S = lambda x: np.log(1.0 + np.square(x))
dS = lambda x: 2.0 * x / (np.square(x) + 1.0)


W = list(
    np.random.random((net_structure[li-1] if li > 0 else input_size, size))*0.1
    for li, size in enumerate(net_structure)
)


Wcp = [w.copy() for w in W]

B = np.random.random((net_structure[-1], net_structure[-2]))*1.0

h = 0.1


lrate = 0.001


# sp_code = False
predictive_output = False



def run(h0, r0, h1, r1, e, fb_factor):
    in0 = x - np.dot(r0, W[0].T)
    top_down0 = e

    h0 += h * (np.dot(in0, W[0]) + fb_factor * np.dot(top_down0, W[1].T))
    
    r0 = act(h0)

    h1 = np.dot(r0, W[1])
    
    r1 = act(h1)
    
    e = y_t - r1

    return h0, r0, h1, r1, e, in0


num_iters, fb_factor, learn = 10000, 1.0, True

# def loop(num_iters, fb_factor, learn):

h0_h = np.zeros((num_iters, batch_size, net_structure[0]))
e0_h = np.zeros((num_iters, batch_size))
h1_h = np.zeros((num_iters, batch_size, net_structure[1]))
d1_h = np.zeros((num_iters, batch_size, net_structure[1]))
e1_h = np.zeros((num_iters, batch_size))
y_h = np.zeros((num_iters, batch_size))
in0_h = np.zeros((num_iters, batch_size, input_size))


h0 = np.zeros((batch_size, net_structure[0]))
r0 = np.zeros((batch_size, net_structure[0]))
h1 = np.zeros((batch_size, net_structure[1]))
r1 = np.zeros((batch_size, net_structure[1]))

e = np.zeros((batch_size, net_structure[-1]))

for i in xrange(num_iters):
    h0, r0, h1, r1, e, in0 = run(h0, r0, h1, r1, e, fb_factor)

    error = np.asarray((
        np.sum(in0 ** 2.0),
        np.sum(e ** 2.0),
    ))
    
    
    if learn:
        # dW0 = np.dot(in0.T, r0)
        dW0 = np.dot(in0.T, np.dot(e, W[1].T))
        dW1 = np.dot(r0.T, e)
        
        W[0] += lrate * dW0
        W[1] += lrate * dW1


    h0_h[i] = h0.copy()
    e0_h[i] = error[0]
    h1_h[i] = h1.copy()
    e1_h[i] = error[1]
    in0_h[i] = in0.copy()
    d1_h[i] = e.copy()
    # return (
    #     h0_h,
    #     e0_h,
    #     h1_h,
    #     e1_h,
    #     in0_h
    # )


# def feedback_perf_curve():
#     r = []
#     fb_factors = np.asarray(list(reversed(1.0-np.log(np.linspace(np.exp(0.0), np.exp(1.0), 100)))))
#     # fb_factors = np.log(np.linspace(np.exp(0.0), np.exp(1.0), 100))
#     # fb_factors = np.linspace(0.0, 1.0, 200)
#     for fb_factor in fb_factors:

#         h0_h, e0_h, h1_h, e1_h, in0_h = loop(100, fb_factor, True)
#         ht0_h, et0_h, ht1_h, et1_h, int0_h = loop(100, 0.0, False)
#         r.append((fb_factor, np.log(et1_h[-1])))
    
#     shs(np.asarray(r))
    

# feedback_perf_curve()


# h0_h, e0_h, h1_h, e1_h, in0_h = loop(200, 1.0, True)
# ht0_h, et0_h, ht1_h, et1_h, int0_h = loop(200, 0.0, False)

# shl(e0_h, et0_h, show=False, title="First layer train/test error")
# shl(e1_h, et1_h, show=False, title="Second layer train/test error")

# shl(h1_h, ht1_h, title="Output test")


