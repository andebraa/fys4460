import pylab as pl
from scipy.ndimage import measurements
from matplotlib.colors import ListedColormap
Lvals = [25,50,100,200,400]
pVals = pl.logspace(pl.log10(0.58), pl.log10(0.85), 20)

C = pl.zeros((len(pVals), len(Lvals)),float)
P = pl.zeros((len(pVals), len(Lvals)),float)
nsamples = 600
mu = pl.zeros(len(Lvals))

for iL in range(len(Lvals)):
    L = Lvals[iL]
    for pIndex in range(len(pVals)):
        p = pVals[pIndex]
        ncount = 0
        perc = []
        while (len(perc)==0):
            ncount = ncount +1
            if (ncount > 1000):
                print('error')
                break
            z = pl.rand(L,L) < p
            lw,num = measurements.label(z)
            perc_x = pl.intersect1d(lw[0,:], lw[-1,:])
            perc = perc_x[pl.where(perc_x > 0)]
        if len(perc) > 0:
            zz  = pl.asarray((lw == perc[0]))
            # zz now contains the spanning cluster
            zzz = zz.T
            #generate band lattice from this
            g = sitetobond(zzz)
            #generate conductivity matrix
            Pvec, c_eff = pl.FIND_COND(g,lx,ly)
            C[pIndex, iL] = C[pIndex, iL] + c_eff
        C[pIndex, iL] = C[pIndex, iL]/nsamples
for iL in range(len(Lvals)):
    L = Lvals[iL]
    pl.plot(pVals, C[:,iL], label='L=' + str(L))
pl.legend()
pl.show()
