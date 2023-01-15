import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt 

x=np.arange(0,10,0.01)
dist = stats.norm(1.5,0.5) 
dist_gaussian = stats.norm.pdf(x,5,np.sqrt(5)) 
dist_poisson = stats.poisson.pmf(x,5) 

plt.figure()
plt.plot(x,dist_gaussian,label='Gaussian')
plt.plot(x,dist_poisson,label='Poisson')
plt.title('PDFs of Poisson and Gaussian Distributions')
plt.legend()
plt.savefig('3.png')

