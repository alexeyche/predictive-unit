
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from util import *
from poc.common import *
from inspect import currentframe, getframeinfo






act_o = Sigmoid()
act = Relu()

np.random.seed(35) 

# lrule = "hebb"
# lrule = "hebb_oja"
# lrule = "hebb_mod"
# lrule = "bp"

input_size = 2
output_size = 1
batch_size = 4
net_size = 30
num_iters = 100
step = 0.1
fb_factor = 0.0
fb_delay = 1

lrate = 0.1

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

xh[10] = x.copy()
xh = smooth_batch_matrix(xh)


yh = np.zeros((num_iters, batch_size, output_size))

yh[10] = y.copy()
yh = smooth_batch_matrix(yh)


W0 = 0.1 - 0.2*np.random.random((input_size, net_size))
b0 = np.zeros((net_size,))

W0 = norm(W0)

# W0r = 0.1 - 0.2*np.random.random((net_size, net_size))

W0r = np.dot(W0.T, W0)

W1 = 0.1 - 0.2*np.random.random((net_size, output_size))
b1 = np.zeros((output_size,))


# u0_given=None
def run(fb_factor, lrate, u0_given=None):
    global W0, W0r, W1, b0, b1


    u0h = np.zeros((num_iters, batch_size, net_size))
    a0h = np.zeros((num_iters, batch_size, net_size))
    u1h = np.zeros((num_iters, batch_size, output_size))
    a1h = np.zeros((num_iters, batch_size, output_size))
    deh = np.zeros((num_iters, batch_size, output_size))
    e0h = np.zeros((num_iters, batch_size, input_size))
    tr0h = np.zeros((num_iters, batch_size, input_size, net_size))
    tr1h = np.zeros((num_iters, batch_size, net_size, output_size))


    u0 = np.zeros((batch_size, net_size)) if u0_given is None else u0_given
    a0 = np.zeros((batch_size, net_size))
    a1 = np.zeros((batch_size, output_size))
    u1 = np.zeros((batch_size, output_size))
    fb0 = np.zeros((batch_size, net_size))
    fb0h = np.zeros((num_iters, batch_size, net_size))

    stop = False
    for i in xrange(num_iters):
        xi = xh[i]
        yi = yh[i]

        x_hat = np.dot(a0, W0.T)
        e0 = xi - x_hat

        # ff = np.dot(xi, W0)
        # rec =  np.dot(a0, W0r)
        
        du0 = (np.dot(e0, W0) + b0) + fb_factor * fb0h[i] - u0
        
        u0 += step * du0

        a0 = act(u0)

        u1 = np.dot(a0, W1) + b1
        a1 = u1
        
        de = yi - a1 
        
        if i < (num_iters-fb_delay):
            fb0h[i+fb_delay] = np.dot(de, W1.T) #* act.deriv(u0)
        
        # for bi in xrange(batch_size):
        #     tr0h[i, bi] = np.outer(e0[bi], a0[bi])
        #     tr1h[i, bi] = np.outer(a0[bi], de[bi])

        W0 += lrate * np.dot(e0.T, a0 - 0.02)
        # W0 += lrate * np.dot((xi - np.dot(a0, W0.T)).T, a0)

        # W0r += 0.05*lrate * np.dot(a0.T, a0)
        # W0 += lrate * np.dot(xi.T, a0)
        

        W1 += lrate * np.dot(a0.T, de)

        W0 = norm(W0)
        
        # b0 += 0.0001*lrate * np.sum(a0,0)   # no bias?
        # b1 += 0.0001*lrate * np.sum(a1,0)

        u0h[i] = u0.copy()
        e0h[i] = e0
        a0h[i] = a0.copy()
        u1h[i] = u1.copy()
        a1h[i] = a1.copy()
        deh[i] = de.copy()
        # fb0h[i] = fb0.copy()    
    
    return (
        u0h,
        e0h,
        a0h,
        u1h,
        a1h,
        deh,
        fb0h,
        stop
    )



for e in xrange(1000):
    u0h, e0h, a0h, u1h, a1h, deh, fb0h, train_stop = run(fb_factor=1.0, lrate=lrate)
    if train_stop:
        break
    ut0h, et0h, at0h, ut1h, at1h, deth, fbt0h, test_stop = run(fb_factor=0.0, lrate=0.0) #, u0_given=u0h[-1].copy())
    if test_stop:
        break
    if e % 100 == 0:
        # shl(u0h[:,3])
        print "{}: E^2 train {:.4f}, E^2 test {:.4f}, {:.4f}".format(e, np.sum(deh ** 2.0), np.sum(deth ** 2.0), np.sum(np.square(at0h - a0h)))


# shl(fb0h[:,1], title="Feedback (b0, train)", show=False)
# shl(et0h[:,1], title="Error layer0 (b0, test)", show=False)

# shl(deh ** 2.0, title="Delta^2 (train)", show=False)
# shl(deth ** 2.0, title="Delta^2 (test)", show=False)
# shl(at1h, title="Output (test)", show=False)

# plt.show()

# shl(ut1h)
