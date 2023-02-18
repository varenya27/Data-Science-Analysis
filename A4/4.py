from scipy import stats 

#a Higgs_Boson p values for 1sigma to 6sigma
p_higgs = [1e-1,1e-2,1e-3,1e-5,1e-7,1e-9]
sig_higgs = stats.norm.isf(p_higgs)
print('Significance for 1sigma to 6sigma:',[ round(sig, 2) for sig in sig_higgs ])

#b lIGO
p_ligo = 2e-7
sig_ligo = stats.norm.isf(p_ligo)
print('Significance of the LIGO discovery:',round(sig_ligo,2))

#c Super-K discovery
chi2,dof = 65.2,67
gof = 1-stats.chi2(dof).cdf(chi2)
print('$\chi^2$ GOF for Super K:',round(gof,2))