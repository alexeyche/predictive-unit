from sympy import *

def inv_matrix(a, b, c, d):
	return Matrix(2, 2, [d/(a*d - b*c), -b/(a*d - b*c), -c/(a*d - b*c),  a/(a*d - b*c)])

w00, w01 = symbols("w00 w01")
W = Matrix(2, 1, (w00, w01))

x = symbols("x")
X = Matrix(1, 1, (x,))


y0, y1 = symbols("y0 y1")
Y = Matrix(2, 1, (y0, y1))

e = X - Matrix(1, 1, [W.T.dot(Y)])

dY = W.dot(e)


A = Matrix(2, 2, [-w00*w00, -w00*w01, -w01*w00, -w01*w01])
b = Matrix(2,1, [w00 * x, w01*x])

Ainv = inv_matrix(*[a for a in A])
