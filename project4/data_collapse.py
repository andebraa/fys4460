import numpy as np
import matplotlib
import pylab
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import measurements
from tqdm import tqdm
np.random.seed(80085)


def matrix_gen(L, p):
    M = np.random.uniform(0,1,(L,L))
    mask = M<p
    M[mask] = 1
    mask = np.logical_not(mask)
    M[mask] = 0
    return M


def find_clusters(array):
    clustered = np.empty_like(array)
    unique_vals = np.unique(array)
    cluster_count = 0
    for val in unique_vals:
        labelling, label_count = ndimage.label(array == val)
        for k in range(1, label_count + 1):
            clustered[labelling == k] = cluster_count
            cluster_count += 1
    return clustered#, cluster_count



def spanning_check(clustered,M):
    spanning=False
    spanning_cluster = 0
    Ms = 0 #the mass of the spanning cluster
    while not spanning:
        for i in range(len(clustered[:,0])):
            for j in range(len(clustered[:,-1])):

                if clustered[i,0] == clustered[j,-1] and M[i,0] != 0: #checks both right and left edge, if the same cluster appears it's bounding
                    empt = np.zeros_like(M)
                    spanning_cluster = clustered[i,0]
                    spanning=True

                    spanning_cluster_matrix = np.where(clustered==spanning_cluster)
                    clustered[spanning_cluster_matrix] = np.max(clustered)+16
                    empt[spanning_cluster_matrix] = 1
                    Ms = np.sum(empt)
                    # norm_inst = matplotlib.colors.Normalize(vmin=None, vmax=None, clip=False)
                    # plt.subplot(2,1,1)
                    # plt.imshow(clustered, norm=norm_inst) #,norm=norm_inst)
                    # plt.subplot(2,1,2)
                    # plt.imshow(empt)
                    # plt.show()
                    return spanning_cluster, spanning, Ms


        return spanning_cluster, spanning, Ms


def data_collapse():
    """
    attempt at re-creating fig 6.2
    """

    L_list = [25,50,100,200]
    p_pc = np.linspace(0, 1, 30) #supposed to be p-pc
    P_list = np.zeros((len(L_list),len(p_pc))) #this is P
    spanns = np.zeros((len(L_list),len(p_pc)))
    iterations = 300
    for i,L in enumerate(L_list):
        for iter in range(iterations): #to get an average for each variable
            spanning = False
            for j,p in enumerate(p_pc):
                m = matrix_gen(L,p)
                #clustered,_ = measurements.label(m)
                clustered = find_clusters(m)
                spanning_cluster, spanning, Ms = spanning_check(clustered,m)
                if spanning:
                    P_list[i,j] += Ms/L**2
                    spanns[i,j] +=1
        P_list[i,:]/=iterations
        spanns[i,:]/=iterations
        D_d = -0.11
        mu = 0.0001

        plt.plot(p_pc*L**(mu), P_list[i,:]*L**(-D_d), label=f'{L}')


    plt.xlabel(r'$(p-p_c)L^x$')
    plt.ylabel(r'y')
    plt.legend()
    plt.show()


def p_vs_L():
    """
    for plotting P(pc,L) to show that P goes to zero for L -> infty
    (see fig 6.1)
    """
    L_list = [2,4,8,16,32,64,128]
    #p_arr = 0.5927
    p = 0.5927
    P_list = np.zeros(len(L_list))
    spanns = np.zeros(len(L_list))
    iterations = 1000
    for i,L in enumerate(L_list):
        for iter in range(iterations): #to get an average for each variable
            spanning = False
            m = matrix_gen(L,p)
            #clustered,_ = measurements.label(m)
            clustered = find_clusters(m)
            spanning_cluster, spanning, Ms = spanning_check(clustered,m)
            if spanning:
                P_list[i] += Ms/L**2
                spanns[i] +=1
        P_list[i]/=iterations
        spanns[i]/=iterations
    plt.plot(L_list, P_list, '-o')
    plt.xlabel('L')
    plt.ylabel('P(pc)')
    plt.show()

def L():
    """
    Generates a P and pi plot vs p for a single size L
    """
    L = 50
    p_arr = np.linspace(0.1, 0.9, 50)
    P_list = np.zeros(len(p_arr))
    spanns = np.zeros(len(p_arr))
    iterations = 10

    for iter in range(iterations): #to get an average for each variable
        spanning = False
        for j,p in enumerate(p_arr):
            m = matrix_gen(L,p)
            #clustered,_ = measurements.label(m)
            clustered = find_clusters(m)
            spanning_cluster, spanning, Ms = spanning_check(clustered,m)
            if spanning:
                P_list[j] += Ms/L**2
                spanns[j] +=1
    P_list/=iterations
    spanns/=iterations
    plt.subplot(2,1,1)
    plt.plot(p_arr, P_list, label=f'{L}')
    plt.xlabel(r'p')
    plt.ylabel(r'P(L,p)')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(p_arr, spanns, label=f'{L}')
    plt.xlabel(r'p')
    plt.ylabel(r"$\Pi(L,p)$")
    plt.legend()


    plt.show()

def clus_num_den(L=200, p=0.5, iterations = 1000, log_base =2, logarit_bins = False):

    iterations = 1000
    L = 10

    tot_area = pylab.array([])
    for i in range(iterations):
        m = matrix_gen(L,p)

        clustered, clustered_number = measurements.label(m)
        labellist = pylab.arange(clustered_number+1)
        area = measurements.sum(m, clustered, labellist)
        tot_area = pylab.append(tot_area, area)

    num, hist = pylab.histogram(tot_area, bins=int(max(tot_area)))
    s = 0.5*(hist[1:]+hist[:-1])
    nsp = num/(L*iterations)

    if not logarit_bins:
        ret = s, nsp

    elif logarit_bins:
        logamax = pylab.ceil(pylab.log(max(s))/(pylab.log(log_base)))
        logbins = log_base**pylab.arange(0,logamax) #this generates a^i bins
        nl, nlbins = pylab.histogram(tot_area, bins=logbins)

        ds = pylab.diff(logbins)
        bins = 0.5*(logbins[1:] + logbins[:-1])
        log_hist_normed = nl/(L**2*ds) #M?
        ret =  bins, log_hist_normed

    return ret

def find_nearest(array, value):
    "stolen from stackexchange"
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def nsp():
    pc = 0.59275
    L_vals = [16, 32, 128, 256]
    n_samples = 1000
    n_bins = 20
    logbase = 10

    for i, L in enumerate(L_vals):
        s, nsp = clus_num_den(L, pc, n_samples, logbase)
        plt.loglog(s, nsp, label=f'L={L}')
        if L == L_vals[-1]:
            start_indx = find_nearest(s,2.6)
            end_indx = find_nearest(s, 8.57)
            tau, b = np.polyfit(np.log(s[start_indx: end_indx]), np.log(nsp[start_indx:end_indx]), deg=1)
            plt.loglog(s, np.exp(b)*s**tau, 'k', label=r'tau: %.3f' %tau)
            tau *= -1
            print(f"tau={tau}")


    plt.xlabel(r's')
    plt.ylabel(r'n(s,p)')
    plt.legend()
    plt.show()


#data_collapse()

nsp()


# m = matrix_gen(30, 0.6)
# plt.subplot(2,1,1)
# plt.imshow(find_clusters(m))
# plt.subplot(2,1,2)
# plt.imshow(find_clusters(m))
# plt.show()
