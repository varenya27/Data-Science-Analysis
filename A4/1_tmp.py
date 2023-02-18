import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats, optimize
import statsmodels
import statsmodels.api as sm

x,y,sigma_y = np.loadtxt('1.txt', unpack=True)
data = np.vstack([x, y, sigma_y])

def polynomial_fit(theta, x):
    """Polynomial model of degree (len(theta) - 1)"""
    return sum(t * x ** n for (n, t) in enumerate(theta))

def logL(theta, model=polynomial_fit, data=data):
    """Gaussian log-likelihood of the model at theta"""
    x, y, sigma_y = data
    y_fit = model(theta, x)
    return sum(stats.norm.logpdf(*args) for args in zip(y, y_fit, sigma_y))

def best_theta(degree, model=polynomial_fit, data=data):
    theta_0 = (degree + 1) * [0]
    neg_logL = lambda theta: -logL(theta, model, data)
    return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)

theta1 = best_theta(1)
theta2 = best_theta(2)
theta3 = best_theta(3)


model = statsmodels.api.OLS(y, x).fit()

#view AIC of model
print(model.aic)



xfit = np.linspace(0, 1)
plt.figure(figsize=(9,7))

plt.errorbar(x,y,sigma_y, fmt='h', ms=6, color='#e27c7c', mfc='#e27c7c', mew=1, ecolor='#a86464', alpha=0.75, capsize=2.0, zorder=0, label='Data');
plt.plot(xfit, polynomial_fit(theta1, xfit), label='best linear model', color='#599e94', ls='--')
plt.plot(xfit, polynomial_fit(theta2, xfit), label='best quadratic model', color='#466964', ls='-.')
plt.plot(xfit, polynomial_fit(theta3, xfit), label='best cubic model', color='#6cd4c5')

plt.legend()
plt.show()