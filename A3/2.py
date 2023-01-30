from scipy.optimize import curve_fit
import numpy as np 
import matplotlib.pyplot as plt 

def line(x, m, c): 
    return c+m*x

ID,x,y,sigma_y,sigma_x,rho_xy = np.loadtxt('data.txt', unpack=True)

#use curve_fit to perform chi^2 minimization
param, param_cov = curve_fit(line, x, y, sigma= sigma_y,absolute_sigma=True)
perr = np.sqrt(np.diag(param_cov))

#printing outputs
m = str(round(param[0],2))
c = str(round(param[1],1))
err_m = str(round(perr[0],2))
err_c = str(round(perr[1],1))
print("m = {} err_m = {}".format(m,err_m))
print('c = {}, err_c = {} '.format(c,err_c))

#plotting the data/line
x_axis = np.linspace(0,300)
plt.figure(figsize=(9,7))
plt.text(125,150,"$y=({}\pm{})x+({}\pm{})$".format(m,err_m,c,err_c),fontsize = 16, fontweight='bold',color='black')
plt.errorbar(x,y,sigma_y, fmt='h', ms=6, color='#115f9a', mfc='#115f9a', mew=1, ecolor='#115f9a', alpha=0.75, capsize=2.0, zorder=0, label='Data');
plt.plot(x_axis, param[0]*x_axis+param[1], '-', color='#a6d75b',label ='Best-fit line')

#minor
plt.title('Data vs. Best Fit Line',fontweight='bold',fontsize=16)
plt.xlim(0,300)
plt.ylim(0,700)
plt.legend(fontsize=16)
plt.savefig('2.png')
