import numpy as np 
from scipy import stats 
from matplotlib import pyplot as plt 

r = np.random.chisquare(3, (1, 10000))
r5 = np.random.chisquare(3, (5, 10000))
r10 = np.random.chisquare(3, (10, 10000))

r = r[:1, :].mean(0)
r5 = r5[:5, :].mean(0)
r10 = r10[:10, :].mean(0)

#figure specifics
fig, ax = plt.subplots(3,1)
fig.suptitle('Central Limit Theorem')
fig.set_figheight(7)
fig.set_figwidth(6)

#plotting the chi2 distributions 
ax[0].hist(r,bins=100,density=True,color='grey',label='$\chi^2$' ) 
ax[1].hist(r5,bins=100,density=True,color='grey',label='5 $\chi^2$ averaged')
ax[2].hist(r10,bins=100,density=True,color='grey',label='10 $\chi^2$ averaged') 

#setting parameters for the normal distributions
mu = 3
sigma = np.sqrt(6/1)
sigma5 = np.sqrt(6/5)
sigma10 = np.sqrt(6/10)

dist = stats.norm(mu, sigma)
dist5 = stats.norm(mu, sigma5)
dist10 = stats.norm(mu, sigma10)

x_pdf = np.linspace(0, 10, 1000)

#plotting the normal distributions
ax[0].plot(x_pdf, dist.pdf(x_pdf), color='black', label='normal pdf')
ax[1].plot(x_pdf, dist5.pdf(x_pdf), color='black', label='normal pdf')
ax[2].plot(x_pdf, dist10.pdf(x_pdf), color='black', label='normal pdf')

#minor
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.savefig('1.png')