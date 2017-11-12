import numpy as np
from util import *


Tmax = 1000.0
dt = 0.1

num = int(Tmax/dt)

# x = np.sin(np.linspace(0, Tmax, num))
x = np.ones(num) * 0.1

W0 = 1.0
W1 = 1.0
fb_factor = 0.0

# Y_t = np.zeros(num)
# Y_t[20] = 1.0
# Y_t = np.sin(np.linspace(0, Tmax, num))

Y_t = np.ones(num) * 0.7

def dYdt(Y, W0, W1, t=0):
	e0 = x[t] - W0 * Y
	e1 = Y_t[t] - W1 * Y
	
	dY = e0 * W0 + fb_factor * e1 * W1

	dW0 = e0 * Y
	dW1 = e1 * Y
	return (e0, e1, dY, dW0, dW1)

Y = 0.0

Y_h = np.zeros(num)
W0_h = np.zeros(num)
W1_h = np.zeros(num)
e0_h = np.zeros(num)
e1_h = np.zeros(num)

for t in xrange(num):
	e0, e1, dY, dW0, dW1 = dYdt(Y, W0, W1, t)

	Y += dt * dY
	W0 += dt * dW0
	W1 += dt * dW1


	Y_h[t] = Y
	W0_h[t] = W0
	W1_h[t] = W1
	e0_h[t] = e0
	e1_h[t] = e1

shl(Y_h, W1_h * Y_h, title="Y, Yo", show=False)
shl(e0_h ** 2.0, e1_h ** 2.0, title="Errors", show=False)
shl(W0_h, W1_h, title="Weights")

# - dW0 is obviously is not correct if we want minimize only the MSE
# - Target:
# 	 L = (x - W0 * Y)^2 + (Yt - W1 * Y)^2*