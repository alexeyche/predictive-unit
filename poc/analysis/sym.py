
import sympy as sp


x, y, yt, W0, W1 = sp.symbols("x y yt W0 W1")

L = ((x - y * W0) ** 2)/2 + ((yt - y * W1) ** 2)/2

Lfixed = ((x - y * W0) ** 2)/2 + ((yt - (x- y*W0) * W0 * W1) ** 2)/2

Lpred_full = ((x - (x - y * W0) * W0) ** 2)/2

Lorig = ((yt - (x * W0) * W1) ** 2)/2

Lpred = ((yt - ( (x - y * W0) * W0) * W1) ** 2)/2


# >>> Lpred.diff(W0)
# (2*W0*W1*y - 2*W1*(-W0*y + x))*(-W0*W1*(-W0*y + x) + yt)/2

# dW0
# (W0*W1*y - W1 * e0) * e1

# dW0'
# y
