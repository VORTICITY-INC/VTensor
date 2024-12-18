import numpy as np

a1 = np.arange(12).reshape(2, 2, 3, order = 'F')

print(a1[..., 0:2])