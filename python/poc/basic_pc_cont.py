
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *

# np.random.seed(5)

act = Linear()
act_o = Linear()

input_size = 1

lrule = Learning.BP


iter_max = 1000

dt = 0.1

x = np.asarray([np.sin(np.linspace(0, dt*iter_max, iter_max) + np.pi)]).T

y_t = np.asarray([np.sin(np.linspace(0, dt*iter_max, iter_max))]).T

net_structure = (1,1)


S = lambda x: np.log(1.0 + np.square(x))
dS = lambda x: 2.0 * x / (np.square(x) + 1.0)

filter_size = 10

W = list(
    np.random.randn(*((net_structure[li-1] if li > 0 else input_size) * filter_size, size))*1.0
    for li, size in enumerate(net_structure)
)


Wcp = [w.copy() for w in W]

B = np.random.random((net_structure[-1], net_structure[-2]))*1.0

fb_factor = 0.0
num_iters = 200
h = 0.01

tau_apical = 2.0
tau_basal = 2.0

lrate = 0.1


# sp_code = False
predictive_output = False


# def loop(num_iters, fb_factor, learn):
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

for i in xrange(num_iters):
    xw = np.concatenate([np.zeros((max(filter_size-i, 0), input_size)), x[max(i-filter_size, 0):i]])
    
    xw_flat = xw.reshape(input_size*filter_size)

    in0 = xw_flat - np.dot(r0, W[0].T)
    top_down0 = e

    h0 += h * (np.dot(in0, W[0]) + fb_factor * np.dot(top_down0, W[1].T))
    
    r0 = act(h0)

    h1 = np.dot(r0, W[1])
    
    r1 = act(h1)

    e = y_t[i] - r1

    error = np.asarray((
        np.sum(in0 ** 2.0),
        np.sum(e ** 2.0),
    ))
    
    
    if learn:
        dW0 = np.outer(in0, r0)
        # dW0 = np.outer(in0, np.dot(e, W[1].T))
        dW1 = np.outer(r0, e)
        
        W[0] += lrate * dW0
        W[1] += lrate * dW1


    h0_h[i] = h0.copy()
    e0_h[i] = error[0]
    h1_h[i] = h1.copy()
    e1_h[i] = error[1]
    in0_h[i] = in0.copy()

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

#         h0_h, e0_h, h1_h, e1_h, in0_h = loop(iter_max, fb_factor, True)
#         ht0_h, et0_h, ht1_h, et1_h, int0_h = loop(iter_max, 0.0, False)
#         r.append((fb_factor, np.log(et1_h[-1])))
    
#     shs(np.asarray(r))
    

# feedback_perf_curve()


# h0_h, e0_h, h1_h, e1_h, in0_h = loop(iter_max, 0.0, True)
# ht0_h, et0_h, ht1_h, et1_h, int0_h = loop(iter_max, 0.0, False)

# shl(e0_h, et0_h, show=False, title="First layer train/test error")
# shl(e1_h, et1_h, show=False, title="Second layer train/test error")

# shl(h1_h, ht1_h, title="Output test")


