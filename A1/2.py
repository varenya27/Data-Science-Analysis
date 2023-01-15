import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt 

x=np.arange(-7,7,0.01)
dist = stats.norm(1.5,0.5) 
dist_gaussian = stats.norm.pdf(x,0,1.5) 
dist_cauchy = stats.cauchy.pdf(x,0,1.5) 

plt.figure()
plt.plot(x,dist_gaussian,label='Gaussian')
plt.plot(x,dist_cauchy,ls='--',label='Cauchy')
plt.title('PDFs of Cauchy and Gaussian Distributions')
plt.legend()
plt.savefig('2.png')

