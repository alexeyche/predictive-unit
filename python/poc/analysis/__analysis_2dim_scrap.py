import numpy as np
from util import *
from poc.common import *


Tmax = 1000.0
dt = 0.1

num = int(Tmax/dt)

act = Relu()

# x = np.sin(np.linspace(0, Tmax, num))
x = np.ones(num) * 0.1

W0 = 0.1
W1 = 0.1
W2 = 0.1
fb_factor = 0.1
lrate = 0.02

# Y_t = np.zeros(num)
# Y_t[20] = 1.0
# Y_t = np.sin(np.linspace(0, Tmax, num))

Y_t = np.ones(num) * 1.0

def dYdt(H, Y, W0, W1, W2, t=0):
	e0 = x[t] - W0 * act(Y)
	fb0 = (W1 * act(Y) - H)

	e1 = act(H) - W1 * act(Y)
	fb1 = Y_t[t] - W2 * act(Y)
	# fb1 = 0.0

	dH = e0 * W0 + fb0 * W1
	dY = e1 * W1 + fb_factor * fb1 * W2
	
	dW0 = e0 * act(H)
	dW1 = e1 * act(Y)
	dW2 = fb1 * act(Y)
	return (e0, e1, fb0, fb1, dH, dY, dW0, dW1, dW2)

H = 0.0
Y = 0.0

Y_h = np.zeros(num)
H_h = np.zeros(num)
W0_h = np.zeros(num)
W1_h = np.zeros(num)
W2_h = np.zeros(num)
e0_h = np.zeros(num)
e1_h = np.zeros(num)
fb0_h = np.zeros(num)
fb1_h = np.zeros(num)

for t in xrange(num):
	e0, e1, fb0, fb1, dH, dY, dW0, dW1, dW2 = dYdt(H, Y, W0, W1, W2, t)

	H += dt * dH
	Y += dt * dY
	W0 += dt * lrate * dW0
	W1 += dt * lrate * dW1
	W2 += dt * lrate * dW2


	Y_h[t] = Y
	H_h[t] = H
	W0_h[t] = W0
	W1_h[t] = W1
	W2_h[t] = W2
	e0_h[t] = e0
	e1_h[t] = e1
	fb0_h[t] = fb0
	fb1_h[t] = fb1


shl(H_h, Y_h, title="Net act", show=False)
shl(Y_t - W2 * Y_h)

# shl(Y_h, W2_h * Y_h, title="Y, Yo", show=False)
# shl(e0_h ** 2.0, e1_h ** 2.0, title="Errors", show=False)
# shl(fb0_h ** 2.0, fb1_h ** 2.0, title="Feedback", show=False)

# shl(W0_h, W1_h, title="Weights")

# - dW0 is obviously is not correct if we want minimize only the MSE
# - Target:
# 	 L = (x - W0 * Y)^2 + (Yt - W1 * Y)^2*