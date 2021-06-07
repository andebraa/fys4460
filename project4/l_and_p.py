from pylab import *
from scipy.ndimage import measurements
"""
Finite size scaling of P(p,L). section 6.2.
This produces mostly the same plot as task3.py function L_and_P
"""
LL = [25, 50, 100, 120,150,180]
p = linspace(0.4,0.75, 50)
nL = len(LL)
nx = len(p)
Ni = zeros(nx)
P = zeros((nx,nL), float)
for iL in range(nL):
    L = LL[iL]
    print(L)
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
                area = measurements.sum(m,lw, perc[0])
                P[ip] = P[ip] + area
    P[:,iL] = P[:,iL]/((L*L)*N)
for iL in range(nL):
    L = LL[iL]
    plot(p,P[:,iL], label=L)
ylabel('P')
xlabel('p(p,L)')
plt.legend()
plt.show()
