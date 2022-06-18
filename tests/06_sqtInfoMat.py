import numpy as np

#couldnt find a way to sqrt a matrix in symforce, so we do it in numpy

A = np.random.normal(size = (3,3))
I = A.T @ A
L = np.linalg.cholesky(I)
R = L.T #<--- sqrt(I)
print(R.T @ R - I)