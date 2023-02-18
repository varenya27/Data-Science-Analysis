import numpy as np 
from scipy import stats 

HIP, Vmag, RA, DE, Plx, pmRA, pmDE, e_Plx, BV=np.loadtxt('HIP_star.dat',skiprows=1, unpack=True)
BV_hyades,BV_nonhyades=[],[]

for i in range(len(HIP)):
    if(50<RA[i]<100 and 0<DE[i]<25 and 90<pmRA[i]<130 and -60<pmDE[i]<-10):
        BV_hyades.append(BV[i])
        continue
    BV_nonhyades.append(BV[i])
t = stats.ttest_ind(BV_hyades, BV_nonhyades)
print('p-value =',t.pvalue)