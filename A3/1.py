import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats 
from astroML.resample import bootstrap
from astroML.stats import median_sigmaG

N = 1000
N_boot = 10000
np.random.seed(10)
dist = stats.norm(0, 1).rvs(N)
sample_boot, sigmaG = bootstrap(dist, N_boot, median_sigmaG, kwargs = dict(axis=1))
x= np.linspace(-1,1,1000)
sigma = np.sqrt(np.pi/(2*N))
pdf = stats.norm(np.mean(sample_boot), sigma).pdf(x)

#plotting the histogram/distribution
plt.figure(figsize=(7,5))
plt.hist(sample_boot, bins=20, density=True, label='Bootstrap samples histogram',color='#48b5c4')
plt.plot(x, pdf,  label='Gaussian fit',color='#115f9a' )

#minor
plt.xlim(-0.3,0.3)
plt.title('Bootstrap Samples')
plt.legend()
plt.savefig('1.png')