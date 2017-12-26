
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

np.random.seed(30)

lrule = "hebb"
# lrule = "hebb_oja"
# lrule = "hebb_mod"
# lrule = "bp"

input_size = 2
output_size = 1
batch_size = 4
net_size = 100
num_iters = 100
step = 0.1

lrate = 0.001

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


W0 = 0.1 - 0.2*np.random.random((input_size, net_size))
b0 = np.zeros((net_size,))

W1 = 0.1 - 0.2*np.random.random((net_size, output_size))
b1 = np.zeros((output_size,))



def simple_dynamics(u0, a0, fb0):
    return (
        ((np.dot(x, W0) + b0) - u0) + fb0
    )

def predictive_dynamics(u0, a0, fb0):
    x_hat = np.dot(a0, W0.T)
    e = x - x_hat
    
    return (
        # ((np.dot(e, W0) + b0) - u0) + fb0
        (np.dot(e, W0) + b0) + fb0
    )


dynamics = simple_dynamics
# dynamics = predictive_dynamics

fb_factor = 1.0
tau_m = 100.0
adapt_gain = 1.0


u0 = np.zeros((batch_size, net_size))
u1 = np.zeros((batch_size, output_size))
ut0 = np.zeros((batch_size, net_size))
ut1 = np.zeros((batch_size, output_size))

dW0 = np.zeros(W0.shape)
db0 = np.zeros(b0.shape)
dW1 = np.zeros(W1.shape)
db1 = np.zeros(b1.shape)

# am0 = np.zeros((batch_size, net_size))
am0 = np.zeros((net_size, ))
amt0 = np.zeros((net_size,))
    
for e in xrange(1001):
    # if e % 10 == 0:
    
    u0 = np.zeros((batch_size, net_size))
    u1 = np.zeros((batch_size, output_size))
    a0 = np.zeros((batch_size, net_size))
    a1 = np.zeros((batch_size, output_size))

    u0h = np.zeros((num_iters, batch_size, net_size))
    a0h = np.zeros((num_iters, batch_size, net_size))
    u1h = np.zeros((num_iters, batch_size, output_size))
    a1h = np.zeros((num_iters, batch_size, output_size))
    
    fb0h = np.zeros((num_iters, batch_size, net_size))

    deh = np.zeros((num_iters, batch_size, output_size))

    W0h = np.zeros((num_iters, input_size, net_size))
    W1h = np.zeros((num_iters, net_size, output_size))
    
    real_bp_signal_h = np.zeros((num_iters, batch_size, net_size))
    proxy_bp_signal_h = np.zeros((num_iters, batch_size, net_size))

    am0h = np.zeros((num_iters, net_size))


    for it in xrange(num_iters): 
        fb0 = fb_factor * np.dot(deh[it-1] if it > 0 else 0.0, W1.T) * act.deriv(u0)

        u0 += step * dynamics(
            u0, 
            a0, 
            fb0
        )
        a0 = act(u0)

        am0 += (adapt_gain*np.mean(a0,0) - am0)/tau_m

        u1 = np.dot(a0, W1) + b1
        # u1 += step * ((np.dot(a0, W1) + b1) - u1)
        a1 = act_o(u1)

        u0h[it] = u0.copy()
        a0h[it] = a0.copy()
        u1h[it] = u1.copy()
        a1h[it] = a1.copy()
        fb0h[it] = fb0.copy()
        am0h[it] = am0.copy()

        de = y - a1 
        deh[it] = de.copy()

        real_bp_signal_h[it] = np.dot(de, W1.T) * act.deriv(u0)
        proxy_bp_signal_h[it] = a0
        # proxy_bp_signal_h[it] = np.dot(x - np.dot(a0, W0.T), W0) * act.deriv(u0)
        
        # derivs
        if lrule == "bp":
            dLdu1 = de * act_o.deriv(u1)

            dW1 = np.dot(a0.T, dLdu1)
            db1 = np.sum(dLdu1, 0)

            dLdu0 = np.dot(de, W1.T) * act.deriv(u0)

            dW0 = np.dot(x.T, dLdu0)
            db0 = np.sum(dLdu0, 0)

        elif lrule == "hebb":

            dLdu1 = de * act_o.deriv(u1)

            dW1 = np.dot(a0.T, dLdu1)
            db1 = np.sum(dLdu1, 0)

            dLdu0 = a0 - am0
            dW0 = np.dot(x.T, dLdu0)
            db0 = np.sum(dLdu0, 0)

        elif lrule == "hebb_oja":

            dLdu1 = de * act_o.deriv(u1)

            dW1 = np.dot(a0.T, dLdu1)
            db1 = np.sum(dLdu1, 0)

            dLdu0 = a0 - am0
            dW0 = np.dot((x - np.dot(a0, W0.T)).T, dLdu0)
            db0 = np.sum(dLdu0, 0)

        elif lrule == "hebb_mod":

            dLdu1 = de * act_o.deriv(u1)

            dW1 = np.dot(a0.T, dLdu1)
            db1 = np.sum(dLdu1, 0)

            dLdu0 = np.dot(x - np.dot(a0, W0.T), W0)
            dW0 = np.dot(x.T, dLdu0)
            
            db0 = np.sum(dLdu0, 0)
        else:
            pass

        # if it == num_iters-1 and e > 10:
        if e > 10:
            W0 += lrate * dW0
            b0 += lrate * db0
            W1 += lrate * dW1
            b1 += lrate * db1

            W0 = norm(W0)

        W0h[it] = W0.copy()
        W1h[it] = W1.copy()

    ut0 = np.zeros((batch_size, net_size))
    ut1 = np.zeros((batch_size, output_size))
    at0 = np.zeros((batch_size, net_size))
    at1 = np.zeros((batch_size, output_size))

    ut0h = np.zeros((num_iters, batch_size, net_size))
    at0h = np.zeros((num_iters, batch_size, net_size))
    ut1h = np.zeros((num_iters, batch_size, output_size))
    at1h = np.zeros((num_iters, batch_size, output_size))

    deth = np.zeros((num_iters, batch_size, output_size))

    for it in xrange(num_iters): 
        ut0 += step * dynamics(
            ut0, 
            at0, 
            np.zeros(ut0.shape)
        )
     
        at0 = act(ut0)

        amt0 += (adapt_gain*np.mean(at0,0) - amt0)/tau_m

        ut1 = np.dot(at0, W1) + b1
        # ut1 += step * (
        #     ((np.dot(at0, W1) + b1) - ut1)
        # )
        at1 = act_o(ut1)

        ut0h[it] = ut0.copy()
        at0h[it] = at0.copy()
        ut1h[it] = ut1.copy()
        at1h[it] = at1.copy()

        det = y - at1 
        deth[it] = det.copy()
        


    if e % 10 == 0:
        print "Epoch {}, train error {:.3f}({:.3f}), test error {:.3f} ({:.3f})".format(
            e, 
            np.linalg.norm(de),
            np.sum((np.round(a1) - y) ** 2.0), 
            np.linalg.norm(det),
            np.sum((np.round(at1) - y) ** 2.0)
        )



# shl(proxy_bp_signal_h[:,1]- am0h, show=False, title="Proxy")
# shl(real_bp_signal_h[:,1], show=True, title="Real")