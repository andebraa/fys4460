import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def matrix_gen(L, p):
    M = np.random.uniform(0,1,(L,L))
    mask = M>p
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
    return clustered, cluster_count

clustered, count = find_clusters(M)
print(clustered)
print(count)

def spanning_check(clustered):
    spanning=False
    spanning_cluster = 0
    while not spanning:
        for i in range(len(clustered[:,0])):
            for j in range(len(clustered[:,-1])):
                if clustered[i,0] == clustered[j,-1] and M[0,i] != 0:
                    print('cunt')
                    spanning_cluster = clustered[i,j]
                    spanning=True

                    spanning_cluster_matrix = np.where(clustered==spanning_cluster)
                    #M[spanning_cluster_matrix] = 99
                    #plt.imshow(clustered)
                    return spanning_cluster, spanning


    return (spanning_cluster, spanning)

spanning_cluster, spanning = spanning_check(clustered)

def main():
    L = 100
    spanning = False
    while not spanning:
main()
