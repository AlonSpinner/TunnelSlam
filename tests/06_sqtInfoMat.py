import numpy as np

A = np.random.normal(size = (3,3))
I = A.T @ A
L = np.linalg.cholesky(I)
R = L.T
print(R.T @ R - I)