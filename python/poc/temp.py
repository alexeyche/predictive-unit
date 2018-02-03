


import numpy as np
W = np.random.random((3, 2))

x = np.random.random((3,))

x_t = np.asarray([1.0, 1.0, 1.0])

y_t = np.asarray([0.5, 0.5])

# for _ in xrange(10):
	
# 	y = np.dot(x, W)

# 	e = y_t - y

	

# 	dW = np.outer(x, e)

# 	W += dW*0.1


# 	print np.linalg.norm(e)



# y = np.dot(x, W)
# for _ in xrange(10):
	

# 	e = y_t - y

# 	y += 0.5*e
	
# 	print np.linalg.norm(e)

y = np.zeros(y_t.shape)
for i in xrange(1000):
	x_hat = np.dot(y, W.T)

	e = x_t - x_hat

	dy = np.dot(e, W)

	y += 0.5*dy
	if i % 100 == 0:
		print np.linalg.norm(e)
