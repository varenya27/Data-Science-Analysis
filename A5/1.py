import numpy as np 
from scipy import stats 
import matplotlib.pyplot as plt 
import pandas as pd

df=pd.read_csv('asteroids.csv')
dens = df['Dens'].to_numpy()
logdens = np.log(dens)

#shapiro-wilk test
shap_dens = stats.shapiro(dens)
shap_logdens = stats.shapiro(logdens)

print('p-value for densities =',shap_dens.pvalue)
print('p-value for log of densities =',shap_logdens.pvalue)

#finding best fit gaussians
mu, std = stats.norm.fit(dens)
mu_log, std_log = stats.norm.fit(logdens)

#plotting everything
fig, ax = plt.subplots(1,2)
fig.set_figwidth(12)
fig.set_figheight(7)
fig.suptitle('Asteroid Densities')
x=np.linspace(-2, 6,10000)
ax[0].hist(dens, bins=7, density=True, label='Asteroid Densities',color='#698269')
ax[0].plot(x,stats.norm.pdf(x,mu,std),label= 'Best-fit gaussian',color='#B99B6B')
ax[0].legend()
ax[1].hist(logdens, bins=4, density=True, label='Log of Asteroid Densities',color='#698269')
ax[1].plot(x,stats.norm.pdf(x,mu_log,std_log),label= 'Best-fit gaussian',color='#B99B6B')
ax[1].legend()
plt.savefig('1.png')