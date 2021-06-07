import numpy as np
import matplotlib
import pylab
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import measurements
from tqdm import tqdm
np.random.seed(80085)
"""
Tools used in task 3 and 4. most of these still exist in task3.py, but i've put
them here for use in 4.
"""


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
