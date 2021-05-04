import numpy as np
import matplotlib
import pylab
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import measurements



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


def L_and_P():
    L_list = [2,4,8,16,32,64,128]
    p_arr = np.linspace(0.1, 0.9, 50)
    P_list = np.zeros((len(L_list),len(p_arr)))
    spanns = np.zeros((len(L_list),len(p_arr)))
    iterations = 1000
    for i,L in enumerate(L_list):
        for iter in range(iterations): #to get an average for each variable
            spanning = False
            for j,p in enumerate(p_arr):
                m = matrix_gen(L,p)
                #clustered,_ = measurements.label(m)
                clustered = find_clusters(m)
                spanning_cluster, spanning, Ms = spanning_check(clustered,m)
                if spanning:
                    P_list[i,j] += Ms/L**2
                    spanns[i,j] +=1
        P_list[i,:]/=iterations
        spanns[i,:]/=iterations
        plt.subplot(2,1,1)
        plt.plot(p_arr, P_list[i,:], label=f'{L}')
        plt.subplot(2,1,2)
        plt.plot(p_arr, spanns[i,:], label=f'{L}')
    plt.subplot(2,1,1)
    plt.xlabel(r'p')
    plt.ylabel(r'P(L,p)')
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel(r'p')
    plt.ylabel(r"$\Pi(L,p)$")
    plt.legend()
    plt.show()

def L():
    L = 500
    p_arr = np.linspace(0.1, 0.9, 50)
    P_list = np.zeros(len(p_arr))
    spanns = np.zeros(len(p_arr))
    iterations = 1000

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

def clus_num_den():
    iterations = 1000
    L = 10
    p = 0.6
    tot_area = []
    for i in range(iterations):
        m = matrix_gen(L,p)

        clustered, clustered_number = measurements.label(m)
        labelList = pylab.arange(clustered_number+1)
        area = measurements.sum(m, clustered, labelList)
        tot_area.append(area)
    num, hist = pylab.histogram(tot_area, bins=int(max(tot_area)))
    s = 0.5*(hist[1:]+hist[:-1])
    nsp = n/(L*nsamp)
    sxi = -1.0/pylab.log(p)
    nsptheory = (1-p)**2*pylab.exp(-s/sxi)
    pylab.plot(s,nsp,'o',s,nsptheory,'-')
    pylab.xlabel('$s$')
    pylab.ylabel('$n(s,p)$')
    pylab.show()



#test()
clus_num_den()
# m = matrix_gen(30, 0.5)
# plt.subplot(2,1,1)
# plt.imshow(find_clusters(m))
# plt.subplot(2,1,2)
# plt.imshow(find_clusters_(m))
