
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *

def norm(data, return_denom=False):
    data_denom = np.sqrt(np.sum(data ** 2))
    data = data/data_denom
    if not return_denom:
        return data
    return data, data_denom


act_o = Sigmoid()
act = Relu()

np.random.seed(35) # 30 is good

# lrule = "hebb"
# lrule = "hebb_oja"
# lrule = "hebb_mod"
# lrule = "bp"

input_size = 2
output_size = 1
batch_size = 4
net_size = 30
num_iters = 1000
step = 0.1
fb_factor = 1.0

lrate = 0.01

x = np.asarray([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

y = np.asarray([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


xh = np.zeros((num_iters, batch_size, input_size))

xh[:] = x.copy()

W0 = 0.1 - 0.2*np.random.random((input_size, net_size))
b0 = np.zeros((net_size,))

W1 = 0.1 - 0.2*np.random.random((net_size, output_size))
b1 = np.zeros((output_size,))



def run(fb_factor, predictive, lrate, u0_given=None):
    global W0, W1, b0, b1
    u0 = np.zeros((batch_size, net_size)) if u0_given is None else u0_given
    u0h = np.zeros((num_iters, batch_size, net_size))

    a0 = np.zeros((batch_size, net_size))
    a0h = np.zeros((num_iters, batch_size, net_size))

    u1 = np.zeros((batch_size, net_size))
    u1h = np.zeros((num_iters, batch_size, net_size))

    a1 = np.zeros((batch_size, net_size))
    a1h = np.zeros((num_iters, batch_size, net_size))

    fb0 = np.zeros((batch_size, net_size))
    fb0h = np.zeros((num_iters, batch_size, net_size))

    deh = np.zeros((num_iters, batch_size, output_size))

    e0h = np.zeros((num_iters, batch_size, input_size))

    for i in xrange(num_iters):
        xi = xh[i]
        
        if predictive:        
            x_hat = np.dot(a0, W0.T)
            e0 = xi - x_hat
        else:
            e0 = xi
        
        du0 = (np.dot(e0, W0) + b0) + fb_factor * fb0

        u0 += step * du0

        a0 = act(u0)


        u1 = np.dot(a0, W1) + b1
        a1 = act_o(u1)
        
        de = y - a1 

        fb0 = np.dot(de, W1.T) #* act.deriv(u0)



        W0 += lrate * np.dot(e0.T, a0)
        # W0 += lrate * np.dot((xi - np.dot(a0, W0.T)).T, a0)
        W1 += lrate * np.dot(a0.T, de)
        
        # b0 -= 0.01 * lrate * np.sum(a0,0)
        # b1 += 0.01 * lrate * np.sum(a1,0)

        u0h[i] = u0.copy()
        e0h[i] = e0.copy()
        a0h[i] = a0.copy()
        u1h[i] = u1.copy()
        a1h[i] = a1.copy()
        deh[i] = de.copy()
        fb0h[i] = fb0.copy()    

    return (
        u0h,
        e0h,
        a0h,
        u1h,
        a1h,
        deh,
        fb0h,
    )


u0h, e0h, a0h, u1h, a1h, deh, fb0h = run(fb_factor=1.0, predictive=True, lrate=lrate)
ut0h, et0h, at0h, ut1h, at1h, deth, fbt0h = run(fb_factor=0.0, predictive=True, lrate=0.0) #, u0_given=u0h[-1].copy())



shl(fb0h[:,1], title="Feedback (b0, train)", show=False)
shl(et0h[:,1], title="Error layer0 (b0, test)", show=False)

shl(deh, title="Delta (train)", show=False)
shl(deth, title="Delta (test)", show=False)



plt.show()


