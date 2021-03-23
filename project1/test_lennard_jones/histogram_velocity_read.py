import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

path = 'dump.lammpstrj'

def readfile(filename):
    """
    Function for reading dump file and printing a histogram of the distances

    args:
        filename (str): path to dump file

    returns:
        histogram_matrix (numpy histogram)
    """
    infile = open(filename, 'r')
    file = infile.readlines()
    num_atoms = int(file[3])

    #extracting header
    header = file[8].split()

    num_bins = 30 #number of bins
    num_timesteps = int(np.shape(file)[0]/(num_atoms+9))
    velocity_values_i = np.zeros(num_atoms)
    print(num_atoms)
    #histogram_matrix = np.zeros((num_timesteps, num_bins, num_bins +1)) #the histogram for each timestep
    histogram_matrix = [] #
    timestep = 0
    i=9
    j=0
    size = int(np.shape(file)[0])
    print(size)
    while i <= (size -1):
        line = file[i]
        if line[:14] == 'ITEM: TIMESTEP':
            timestep +=1
            histogram_matrix.append(np.histogram(velocity_values_i, bins=num_bins))
            i += 9

        elem = line.split()
        print(i, timestep, elem[5:], j - (num_atoms+9)*timestep)
        j = (i-(9+ 9*timestep))
        print(j - ((num_atoms)*timestep))
        velocity_values_i[j - ((num_atoms)*timestep)] = np.linalg.norm(elem[5:])
        i +=1

    infile.close()
    return histogram_matrix

# def maxwell(x):
#     return 600*np.sqrt(2/np.pi)*x**2 * np.exp((-(1.5*x)**2)/2)

histogram_matrix= readfile(path)
norm = np.dot(histogram_matrix[-1][0], histogram_matrix[-1][0])
dots = np.zeros(len(histogram_matrix))
for i in range(1,len(histogram_matrix)):
    dots[i] = np.dot(histogram_matrix[-1][0], histogram_matrix[i][0])/(norm)
plt.plot(dots)
plt.xlabel('steps /10')
plt.ylabel('$\sum_i h_i (t) h_i(t_n)/ \sum_i h_i(t_n) h_i(t_n)$')
plt.show()

plt.subplot(211)
plt.title('initial state')
plt.bar(histogram_matrix[0][1][:-1], histogram_matrix[0][0])
plt.xlabel('velocity')
plt.subplot(212)
plt.title('final state')
plt.xlabel('velocity')
plt.bar(histogram_matrix[-1][1][:-1], histogram_matrix[-1][0])
#attempt at plotting maxwell boltzmann distribution. not worth the time
# plt.plot(np.linspace(0,7,100), maxwell(np.linspace(0,7,100)))
# plt.plot(np.linspace(0,7,100), maxwell(np.linspace(0,7,100)), '-r')
# plt.bar(histogram_matrix[-1][1][:-1], histogram_matrix[-1][0])
plt.show()
