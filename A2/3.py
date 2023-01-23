import numpy as np 
from scipy import stats 
from matplotlib import pyplot as plt 

#generating/extracting the data
wind_data = np.loadtxt('3.txt',unpack=True)
x=np.arange(0,20,1)
dist=stats.dweibull(2,0,6)
weibull= 200*dist.pdf(x) 

#plotting
plt.bar(x,wind_data,color='lightgrey',label= 'Wind Speed Frequencies')
plt.plot(x,weibull,color='black',label='Weibull Distribution, $k=2, \lambda=6$')

#minor
plt.legend()
plt.title('Wind Speeds')
plt.xlabel('wind speeds (m/s)')
plt.ylabel('freq (%)')
plt.savefig('3.png')