import numpy as np 
from scipy import stats 

N = 50 #from the source code
chi2 = np.array([0.96,0.24,3.84,2.85])*(N-1) 
p =  1 - stats.chi2.cdf(chi2,N-1)
print('p-values =',p)