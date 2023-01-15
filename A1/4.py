import numpy as np

k = np.array([0.8920, 0.881, 0.8913, 0.9837, 0.8958])
e = np.array([0.00044,0.009,0.00032,0.00048,0.00045])

k_weighted = np.sum(k/e**2)/np.sum(1/e**2)
e_weighted = np.sqrt(1/np.sum(1/e**2))

print(k_weighted, e_weighted)

