import numpy as np 
from scipy import stats 
from matplotlib import pyplot as plt 

L,z = np.loadtxt('2.txt', unpack=True)

corr_coeff,p_pearson = stats.pearsonr(z,L) 
rho,p_spearman = stats.spearmanr(z,L) 
tau,p_kendall = stats.kendalltau(z,L)

print('pearson correlation coefficient, p-value=',corr_coeff,p_pearson)
print('spearmans rho, p-value=',rho,p_spearman)
print('kenalls tau coefficient, p-value=',tau, p_kendall)

plt.plot(np.log10(z), np.log10(L),'.',color='grey')
plt.title('Luminosity v. Redshift')
plt.xlabel('$log(z)$')
plt.ylabel('$log(L_x)$')
plt.grid()
plt.savefig('2.png')