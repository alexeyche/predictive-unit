
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *

np.random.seed(5)

act = Linear()
act_o = Linear()

input_size = 20

lrule = Learning.HEBB
# lrule = Learning.HEBB

x = np.random.random(input_size)


y_t = np.asarray([0.1, 0.8, 0.3])

net_structure = (10, 5, 3)


B = list(
    np.random.random((net_structure[-1], size))*1.0
    for li, size in enumerate(net_structure[:-1])
)

W0 = list(
    np.random.random((net_structure[li-1] if li > 0 else input_size, size))*0.5
    for li, size in enumerate(net_structure)
)


# fb_factor = 0.0
# nudge_factor = 0.0
# num_iters = 1000
# step = 0.05
# lrate = 0.05
# decay_rate = 0.0
# tau_m = 5.0
# adapt_gain = 1.0


def run(
    fb_factor = 1.0,
    nudge_factor = 1.0,
    num_iters = 1000,
    step = 0.05,
    lrate = 0.01,
    decay_rate = 0.0,
    tau_m = 5.0,
    adapt_gain = 1.0,
    W = None
):
    W = [w.copy() for w in W]

    h = [np.zeros(ns) for ns in net_structure]
    e = [np.zeros(ns) for ns in (input_size, ) + net_structure[:-2] + net_structure[-1:]] 
    r = [np.zeros(ns) for ns in net_structure]
    rm = [np.zeros(ns) for ns in net_structure]


    h_h = [np.zeros((num_iters, ns)) for ns in net_structure]
    e_h = [np.zeros((num_iters, ns)) for ns in (input_size, ) + net_structure[:-2] + net_structure[-1:]]
    r_h = [np.zeros((num_iters, ns)) for ns in net_structure]
    rm_h = [np.zeros((num_iters, ns)) for ns in net_structure]
    err_h = np.zeros((num_iters, len(net_structure)))

    for i in xrange(num_iters):

        e[0] = x - np.dot(r[0], W[0].T)
        h[0] += step * (np.dot(e[0], W[0]) - fb_factor*e[1])
        r[0] = act(h[0]) # - rm[0])
        
        # rm[0] += (adapt_gain*r[0] - rm[0])/tau_m
        

        e[1] = r[0] - np.dot(r[1], W[1].T)
        h[1] += step * (np.dot(e[1], W[1])- nudge_factor*np.dot(e[2], W[2].T))
        r[1] = act(h[1]) # - rm[1])
        
        # rm[1] += (adapt_gain*r[1] - rm[1])/tau_m
        
        e[2] = np.dot(r[1], W[2]) - y_t

        ###

        error = tuple(np.linalg.norm(ee) for ee in e)

        for h_hl, hl, r_hl, rl, rm_hl, rml in zip(h_h, h, r_h, r, rm_h, rm):
            h_hl[i] = hl.copy()
            r_hl[i] = rl
            rm_hl[i] = rml

        for e_hl, el in zip(e_h, e):
            e_hl[i] = el.copy()

        err_h[i] = np.asarray(error)
        
        if lrule == Learning.BP:
            da1 = np.dot(-e[2], W[2].T)

            dW = [
                np.outer(e[0], np.dot(da1, W[1].T)),
                np.outer(e[1], da1),
                np.outer(r[-2], -e[2]),
            ]
        else:
            dW = []
            for ee, rr in zip(e[:-1], r[:-1]):
                dW.append(np.outer(ee, rr))
            
            dW.append(np.outer(r[-2], -e[2]))
            
        # dW[-1] = np.zeros(W[-1].shape)


        for li in xrange(len(net_structure)):
            W[li] += lrate * dW[li]

    return h_h, r_h, rm_h, e_h, err_h



h_h, r_h, rm_h, e_h, err_h = run(lrate=0.05, nudge_factor=1.0, fb_factor=1.0, W=W0)


# ht_h, rt_h, rmt_h, et_h, err_t_h = run(lrate=0.0, nudge_factor=0.0, step=0.1)