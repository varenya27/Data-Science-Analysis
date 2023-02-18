import numpy as np 
from matplotlib import pyplot as plt 
from scipy import stats 
from sklearn.mixture import GaussianMixture

n_comp = np.arange(1,11)
t90 = np.loadtxt('beppoSax.txt',unpack=True)
data =np.log10(t90)
aic,bic,models=([] for _ in range(3))
for n in n_comp:
    gmm = GaussianMixture(n_components = n)
    X=np.expand_dims(data,1)
    gmm = gmm.fit(X)
    models.append(gmm)
    aic.append(gmm.aic(X))
    bic.append(gmm.bic(X))
print('Number of components =',n_comp)
print("AIC =",[ round(elem, 2) for elem in aic ])
print("BIC =",[ round(elem, 2) for elem in bic ])
print('opitimum n =',n_comp[np.argmin(aic)])

i_plot=[1,4,8]
for i in i_plot:
    x = np.linspace(-2,4,1000)
    y = np.exp(models[i].score_samples(x.reshape(-1,1)))
    plt.figure(figsize=(7,10))
    plt.plot(x, y, color="#B99B6B", lw=3, label='GMM')
    plt.hist(data, bins=20, density=True,color='#698269')
    plt.title('GMM for {} components'.format(n_comp[i]))
    plt.ylabel('N')
    plt.xlabel('log$_{10}$(T90)')
    plt.legend()
    plt.savefig('3gmm_{}'.format(str(i+1))+'.png')

#plotting the aic and bic values
plt.figure(figsize=(9,7))
plt.plot(n_comp,aic,label='AIC',ls='--',color='#B99B6B',lw=3)
plt.plot(n_comp,bic,label='BIC',color='#B99B6B',lw=3)
plt.scatter(i+1,bic[i],c='#B99B6B',s=50,)
plt.scatter(i+1,aic[i],c='#B99B6B',s=50)
plt.ylim(1840,2030)
plt.xlabel('number of components')
plt.ylabel('AIC/BIC values')
plt.title('AIC and BIC values for different numbers of components in the GMM')
plt.legend()
plt.savefig('3_aic.png')