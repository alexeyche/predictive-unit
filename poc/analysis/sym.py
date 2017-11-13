
import sympy as sp


x, y, yt, W0, W1 = sp.symbols("x y yt W0 W1")

L = ((x - y * W0) ** 2)/2 + ((yt - y * W1) ** 2)/2

Lfixed = ((x - y * W0) ** 2)/2 + ((yt - (x- y*W0) * W0 * W1) ** 2)/2

Lpred_full = ((x - (x - y * W0) * W0) ** 2)/2

Lorig = ((yt - (x * W0) * W1) ** 2)/2

Lpred = ((yt - ( (x - y * W0) * W0) * W1) ** 2)/2








# Lpred0 = ((yt - ( (x - y * W0) * W0) * W1) ** 2)/2
Lpred = ((yt - (x * W0 - y * W0 * W0) * W1) ** 2)/2
