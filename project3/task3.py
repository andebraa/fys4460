import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage


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
    while not spanning:
        for i in range(len(clustered[:,0])):
            for j in range(len(clustered[:,-1])):

                if clustered[i,0] == clustered[j,-1] and M[i,0] != 0:
                    print('cunt')
                    spanning_cluster = clustered[i,j]
                    spanning=True

                    spanning_cluster_matrix = np.where(clustered==spanning_cluster)
                    clustered[spanning_cluster_matrix] = np.max(clustered)+10
                    #map = matplotlib.colors.Colormap('name')
                    plt.imshow(clustered) #, cmap=map)
                    plt.show()
                    return spanning_cluster, spanning


        return (spanning_cluster, spanning)


def main():
    L = 100
    spanning = False
    p = 0.5
    while not spanning:
        print(p)
        m = matrix_gen(L,p)
        clustered=find_clusters(m)
        spanning_cluster, spanning = spanning_check(clustered,m)
        p+=0.001
    print(m)
    print(spanning_cluster, spanning)
main()
