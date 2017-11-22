import numpy as np
from util import *


# x = np.sin(np.linspace(0, Tmax, num))

fb_factor = 1.0
net_lrate = 1.0
out_lrate = 0.0
ff_factor = 1.0

# Y_t = np.zeros(num)
# Y_t[20] = 1.0
# Y_t = np.sin(np.linspace(0, Tmax, num))

x = 0.5
Y_t = 0.7


def dYdt(Y, W0, W1):
    e0 = x - W0 * Y
    e1 = Y_t - W1 * Y
    
    dY = ff_factor * e0 * W0 + fb_factor * e1 * W1

    dW0 = e0 * Y
    dW1 = e1 * Y
    return (e0, e1, dY, dW0, dW1)


def run(num, Y=0.0, W0=1.0, W1=1.0):
    Y_h = np.zeros(num)
    W0_h = np.zeros(num)
    W1_h = np.zeros(num)
    e0_h = np.zeros(num)
    e1_h = np.zeros(num)

    for t in xrange(num):
        e0, e1, dY, dW0, dW1 = dYdt(Y, W0, W1)

        Y += dt * dY
        W0 += dt * net_lrate * dW0
        # W1 += dt * out_lrate * dW1

        
        Y_h[t] = Y
        W0_h[t] = W0
        W1_h[t] = W1
        e0_h[t] = e0
        e1_h[t] = e1
    
    return Y_h, W0_h, W1_h, e0_h, e1_h
    


def plot_flow_dY_dW0(
    nb_points,
    Y_range, 
    W0_range,
    normalize=False,
):
    Y_min, Y_max = Y_range
    W0_min, W0_max = W0_range

    Y_a = np.linspace(Y_min, Y_max, nb_points)
    W0_a = np.linspace(W0_min, W0_max, nb_points)


    dY = np.zeros((nb_points, nb_points))
    dW0 = np.zeros((nb_points, nb_points))
    E = np.zeros((nb_points, nb_points))
    
    Y_mg, W0_mg = np.zeros((nb_points, nb_points)), np.zeros((nb_points, nb_points))
    
    for yy_id, yy in enumerate(Y_a):
        for ww_id, ww in enumerate(W0_a):
            e0r, e1r, dYr, dW0r, dW1r = dYdt(Y=yy, W0=ww, W1=1.0)

            dY[yy_id, ww_id] = dYr
            dW0[yy_id, ww_id] = dW0r
            E[yy_id, ww_id] = e1r ** 2
            Y_mg[yy_id, ww_id] = yy
            W0_mg[yy_id, ww_id] = ww

    M = np.hypot(dY, dW0)
    
    if normalize:
        M[M == 0] = 1. 

        dY /= M
        dW0 /= M

    plt.figure(figsize=(7,7))
    plt.title('Trajectories and direction fields')
    Q = plt.quiver(
        Y_mg, 
        W0_mg, 
        dY, 
        dW0, 
        M, 
        pivot='mid', 
        cmap=plt.cm.seismic,  #cmap=plt.cm.jet
        headlength=7, 
        headwidth=5.0
    )
    plt.xlabel('Y')
    plt.ylabel('W0')
    plt.legend()
    plt.grid()
    plt.xlim(Y_min, Y_max)
    plt.ylim(W0_min, W0_max)
    plt.show()

    return dY, dW0, E, M

dY, dW0, E, M = plot_flow_dY_dW0(30, (-2.0, 2.0), (-2.0, 2.0))

# Tmax = 100.0
# dt = 0.1

# num = int(Tmax/dt)


# Y_h, W0_h, W1_h, e0_h, e1_h = run(num, Y=0.0, W0=0.5, W1=1.0)

# shl(Y_h, W1_h * Y_h, title="Y, Yo", show=False)
# shl(e0_h ** 2.0, e1_h ** 2.0, title="Errors", show=True)



# shl(W0_h, W1_h, title="Weights")

# - dW0 is obviously is not correct if we want minimize only the MSE
# - Target:
#      L = (x - W0 * Y)^2 + (Yt - W1 * Y)^2*