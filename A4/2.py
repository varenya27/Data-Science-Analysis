import numpy as np 
from scipy import stats, optimize

data = np.array([[ 0.42,  0.72,  0.  ,  0.3 ,  0.15,
                   0.09,  0.19,  0.35,  0.4 ,  0.54,
                   0.42,  0.69,  0.2 ,  0.88,  0.03,
                   0.67,  0.42,  0.56,  0.14,  0.2  ],
                 [ 0.33,  0.41, -0.22,  0.01, -0.05,
                  -0.05, -0.12,  0.26,  0.29,  0.39, 
                   0.31,  0.42, -0.01,  0.58, -0.2 ,
                   0.52,  0.15,  0.32, -0.13, -0.09 ],
                 [ 0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                   0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                   0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                   0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1  ]])

def polynomial_fit(theta, x):
    return sum(t * x ** n for (n, t) in enumerate(theta))
def best_theta(degree, model=polynomial_fit, data=data):
    theta_0 = (degree + 1) * [0]
    neg_logL = lambda theta: -logL(theta, model, data)
    return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)
def logL(theta, model=polynomial_fit, data=data):
    """Gaussian log-likelihood of the model at theta"""
    x, y, sigma_y = data
    y_fit = model(theta, x)
    return sum(stats.norm.logpdf(*args)
               for args in zip(y, y_fit, sigma_y))
def aic(data,theta,model=polynomial_fit):
    deg=len(theta)
    aic =-2*logL(theta,model,data)+2*deg
    return aic
def bic(data,theta,N,model=polynomial_fit):
    k=len(theta)
    bic= -2*logL(theta,model,data)+k*np.log(N)
    return bic


x, y, sigma_y = data
N=len(x)
theta1 = best_theta(1)
theta2 = best_theta(2)

print('\n-AIC Values-')
print('Linear:',aic(data,theta1))
print('Quad:',aic(data,theta2))

print('\n-BIC Values-')
print('Linear:',bic(data,theta1,N))
print('Quad:',bic(data,theta2,N))
