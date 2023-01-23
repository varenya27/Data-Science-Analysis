import numpy as np 
from scipy import stats 

a=np.arange
x = stats.norm(0,1).rvs(1000)
y = stats.norm(0,1).rvs(1000)

r,p_pearson = stats.pearsonr(x,y) 
t = r*np.sqrt(998/(1-r**2))

if t>0:
    p_t = 2*(1-stats.t.cdf(t,998))
else:
    p_t = 2*(stats.t.cdf(t,998))

print('pearson coefficient:',r,'\np-value:',p_pearson)
print('p-value from Students t dist:',p_t)