import pylab as pl
from scipy.ndimage import measurements
from matplotlib.colors import ListedColormap
from scipy.sparse import spdiags, dia_matrix, coo_matrix
from scipy.sparse.linalg import spsolve

Lvals = [25,50,100,200,400]
pVals = pl.logspace(pl.log10(0.58), pl.log10(0.85), 20)

C = pl.zeros((len(pVals), len(Lvals)),float)
P = pl.zeros((len(pVals), len(Lvals)),float)
nsamples = 600
mu = pl.zeros(len(Lvals))

def sitetobond(z):
    #Function to convert the site network z(L,L) into a (L*L,2) bond
    #network
    # g [i,0] gives bond perpendicular to direction of flow
    # g [i,1] gives bond parallel to direction of flow
    # z [ nx, ny] -> g [nx * ny, 2]
    nx = pl.size(z, 1-1)
    ny = pl.size(z, 2-1)
    N = nx*ny
    gg_r = pl.zeros((nx,ny)) #first find these
    gg_d = pl.zeros((nx,ny))
    gg_r[:,0:ny-1] = z[:,0:ny-1]*z[:,1:ny]
    gg_r[:,ny-1] = z[:,ny-1]

    gg_d[0:nx-1,:] = z[0:nx-1,:]*z[1:nx,:]
    gg_d[nx-1,:] = 0
    #then concatenate gg onto g
    g = pl.zeros((nx*ny,2))
    g[:,0] = gg_d.reshape(-1,order='F').T
    g[:,1] = gg_r.reshape(-1,order='F').T
    return g
def FIND_COND(A,X,Y):
    V_in = 1.
    V_out = 0.
    #calls MK_Eqsystem
    B,C = MK_EQSYSTEM(A,X,Y)
    #Kirchhoff's equations solve for V
    V  = spsolve(B,C)
    #the pressure at the external sites is added
    #(boundary condition)
    V = concatenate((V_in*pl.ones(X), V,V_out*pl.ones(X)))
    #calculate ceff
    #second-last X elements of V multiplied with second-last elem.of A
    #these are the second last column of the system
    #gives the conductivity of the system per row?
    Ceff = pl.dot((V[-1-2*X:-1-X] -V_out).T, A[-1-2*X:-1-X], 1))\
    /(V_in - V_out)
    return V,Ceff

def MK_EQSYSTEM(A,X,Y):
    #total no of internal lattice sites
    sites = X*(Y-2)
    #allocate space for the nonzeros uppder diagonals
    main_diag = pl.zeros(sites)
    uppder_diag1 = pl.zeros(sites-1)
    uppder_diag2 = pl.zeros(sites-1)
    #calculate the nonzero upper diagonals
    main_diag = A[X:X*(Y-1), 0] + A[X:X*(Y-1),1] +\
                A[0:X*(Y-2),1] + A[X-1:X*(Y-1)-1,0]
    upper_diag1 = A[X:X*(Y-1)-1,0]
    upper_diag2 = A[X:X*(Y-2),1]
    main_diag[where(main_diag==0)] = 1
    #constructing B which is symmetric, lower = upper diagonals
    B = dia_matrix((sites,sites)) #B*u=t
    B = B + -spdiags(upper_diag2, -X, sites, sites)
    B = B+B.T + spdiags(main_diag, 0 , sites, sites )
    #constructing C
    C = pl.zeros(sites)
    # C = dia_matrix ( (sites, 1))
    C[0:X] = A[0:X, 1]
    C[-1-X+1:-1] = 0*A[-1-2*x+1:-1-X,1]
    return B,C


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
