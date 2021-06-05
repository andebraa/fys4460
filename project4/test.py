from pylab import *
from scipy.ndimage import measurements
LL = [25, 50, 100, 200]
p = linspace(0.4,0.75, 50)
nL = len(LL)
nx = len(p)
P = zeros((nx,nL), float)
for iL in range(nL):
    L = LL[iL]
    N = int(2000*25/L)
    for i in range(N):
        z = rand(L,L)
        for ip in range(nx):
            m = z<p[ip]
            lw, num = measurements.label(m)
            perc_x = intersect1d(lw[0,:], lw[-1,:])
            perc = perc_x[where(perc_x >0)]
            if (len(perc)>0):
                Ni[ip] = Ni[ip] +1
                area = measurments.sum(m,lw, perc[0])
                P[ip] = P[ip] + area
    P[:,iL] = P[:,iL]/((L*L)*N)
for iL in range(nL):
    L = LL[iL]
    plot(p,P[:,iL])
    ylabel('P')
    xlabel(p)
