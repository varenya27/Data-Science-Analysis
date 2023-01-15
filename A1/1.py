import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt 

x=np.arange(0,3,0.01)
n = stats.norm.pdf(x,1.5,0.5) 
plt.figure()
plt.plot(x,n,label='normal dist')
plt.title('Normal Distribution PDF')
plt.legend()
plt.savefig('1.png')

dist = stats.norm(1.5,0.5) 
sample = dist.rvs(1000)

mean = np.mean(sample)
variance = np.var(sample)
skewness = stats.skew(sample)
kurt = len(sample)*np.sum( (sample-np.mean(sample))**4 )/(np.sum((sample-np.mean(sample))**2))**2
MAD = np.median(np.abs(sample-np.median(sample)))
std_dev_mad = 1.482 * MAD

sigma_g =  0.7413 * (np.percentile(sample,75)-np.percentile(sample,25)) 

print('mean =',mean)
print('variance =',variance)
print('skewness =',skewness)
print('kurtosis =',kurt)
print('Standard Deviation using MAD =',std_dev_mad)
print('Sigma_g =',sigma_g)