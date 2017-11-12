
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *

# np.random.seed(5)

act = Linear()
act_o = Linear()

input_size = 3

x = np.random.random(input_size)


y_t = np.asarray([0.5])

net_structure = (5, 2, 1)


W = list(
    np.random.random((net_structure[li-1] if li > 0 else input_size, size))*0.5
    for li, size in enumerate(net_structure)
)


Wcp = [w.copy() for w in W]

B = list(
    np.random.random((net_structure[-1], size))*1.0
    for li, size in enumerate(net_structure[:-1])
)


fb_factor = 1.0
nudge_factor = 1.0
num_iters = 1000

step = 0.1

tau_apical = 2.0
tau_basal = 2.0

lrate = 0.0
decay_rate = 0.0

h = [np.zeros(ns) for ns in net_structure]
e = [np.zeros(ns) for ns in (input_size, ) + net_structure[:-1]]
r = [np.zeros(ns) for ns in net_structure]


h_h = [np.zeros((num_iters, ns)) for ns in net_structure]
e_h = [np.zeros((num_iters, ns)) for ns in (input_size, ) + net_structure[:-1]]
r_h = [np.zeros((num_iters, ns)) for ns in net_structure]
err_h = np.zeros((num_iters, len(net_structure)))

for i in xrange(num_iters):

    e[0] = x - np.dot(r[0], W[0].T)
    h[0] += step * (np.dot(e[0], W[0]) - fb_factor*e[1])
    r[0] = act(h[0])


    e[1] = r[0] - np.dot(r[1], W[1].T)
    h[1] += step * (np.dot(e[1], W[1])) - nudge_factor*e[2]
    r[1] = act(h[1])

    e[2] = np.dot(r[1], W[2]) - y_t


    ###

    error = tuple(np.linalg.norm(ee) for ee in e)

    for h_hl, hl, r_hl, rl in zip(h_h, h, r_h, r):
        h_hl[i] = hl.copy()
        r_hl[i] = rl

    for e_hl, el in zip(e_h, e):
        e_hl[i] = el.copy()

    err_h[i] = np.asarray(error)
    
    dW = []
    for ee, rr in zip(e[:-1], r[:-1]):
        dW.append(np.outer(ee, rr))
    
    dW.append(np.outer(r[-2], -e[2]))
    
    for li in xrange(len(net_structure)):
        W[li] += lrate * dW[li]
