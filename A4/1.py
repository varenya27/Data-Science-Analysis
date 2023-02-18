import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats, optimize

x,y,sigma_y = np.loadtxt('1.txt', unpack=True)
data = np.vstack([x, y, sigma_y])
N = len(x)+1



def polynomial_fit(theta, x):
    return sum(t * x ** n for (n, t) in enumerate(theta))

def logL(theta, model=polynomial_fit, data=data):
    x, y, sigma_y = data
    y_fit = model(theta, x)
    return sum(stats.norm.logpdf(*args) for args in zip(y, y_fit, sigma_y))

def best_theta(degree, model=polynomial_fit, data=data):
    theta_0 = (degree + 1) * [0]
    neg_logL = lambda theta: -logL(theta, model, data)
    return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)

def aic(data,theta,model=polynomial_fit):
    deg=len(theta)
    aic =-2*logL(theta,model,data)+2*deg
    return aic
def bic(data,theta,N,model=polynomial_fit):
    k=len(theta)
    bic= -2*logL(theta,model,data)+k*np.log(N)
    return bic
theta1 = best_theta(1)
theta2 = best_theta(2)
theta3 = best_theta(3)

print('-Best fit parameters (lowest coefficient first)- ')
print('Linear:',theta1)
print('Quad:',theta2)
print('Cubic:',theta3)
print('\n-Maximum Likelihoods-')
print('Linear:', logL(best_theta(1)))
print('Quad:', logL(best_theta(2)))
print('Cubic:', logL(best_theta(3)))

print('\n-AIC Values-')
print('Linear:',aic(data,theta1))
print('Quad:',aic(data,theta2))
print('Cubic:',aic(data,theta3))

print('\n-BIC Values-')
print('Linear:',bic(data,theta1,N))
print('Quad:',bic(data,theta2,N))
print('Cubic:',bic(data,theta3,N))

#chi2 values
chi2_lin = np.sum(((y-theta1[0]-theta1[1]*x)/sigma_y)**2)
chi2_quad = np.sum(((y-theta2[0]-theta2[1]*x-theta2[2]*x**2)/sigma_y)**2)
chi2_cub = np.sum(((y-theta3[0]-theta3[1]*x-theta3[2]*x**2-theta3[3]*x**3)/sigma_y)**2)
p_lin_quad =1-stats.chi2(3-2).cdf(np.abs(chi2_quad-chi2_lin))
p_lin_cub =1-stats.chi2(4-2).cdf(np.abs(chi2_cub-chi2_lin))
print('\n-p-values wrt linear fit-')
print('Quad:',p_lin_quad)
print('Cubic:',p_lin_cub)
#plots
xfit = np.linspace(0, 1)
plt.figure(figsize=(9,7))

plt.errorbar(x,y,sigma_y, fmt='h', ms=6, color='#fd7f6f', mfc='#fd7f6f', mew=1, ecolor='#fd7f6f', alpha=0.75, capsize=2.0, zorder=0, label='Data');
plt.plot(xfit, polynomial_fit(theta1, xfit), label='best linear model', color='#7eb0d5', ls='--')
plt.plot(xfit, polynomial_fit(theta2, xfit), label='best quadratic model', color='#b2e061', ls='-.')
plt.plot(xfit, polynomial_fit(theta3, xfit), label='best cubic model', color='#bd7ebe')

plt.legend()
plt.savefig('1.png')