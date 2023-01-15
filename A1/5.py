import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#reading the data
data = pd.read_csv("exoplanet.eu_catalog.csv")
e=data['eccentricity'].dropna().tolist()

#considering only the positive values (as is required for boxcox)
e = [x for x in e if x != 0]
e_norm,lamb = stats.boxcox(e)

print(e,e_norm) #printing out the samples

#plotting the histograms
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figheight(6)
fig.set_figwidth(8)
fig.suptitle('Eccentricites')
ax1.hist(e, 50, density=False)
ax1.set_title('Eccentricity Dist.')
ax1.set_xlim(0,1)
ax2.hist(e_norm, 50, density=False)
ax2.set_title('Gaussianized Eccentricity Dist.')
plt.savefig('5.png')