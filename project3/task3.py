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
M = matrix_gen(10,0.1)
print(M)

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
def spanning_ckeck(clustered):
    spanning = False
    while not spanning:
        for i in range(len(clustered[0,:])):
            for j in range(len(clustered[-1,:])):
                if clustered[0,i] == clustered[-1,j] and M[0,i] != 0:
                    spanning_cluster = M[i,j]
                    print(spanning_cluster)
                    spanning=True
                    return spanning_cluster

        spanning = False
        return spanning_cluster

if spanning:
    spanning_cluster_matrix = np.where(clustered==spanning_cluster)
    #M[spanning_cluster_matrix] = 99
    #plt.imshow(clustered)
    plt.imshow(M)
    plt.show()
