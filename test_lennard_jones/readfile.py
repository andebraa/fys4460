import numpy as np
import matplotlib.pyplot as plt

path = 'dump.lammpstrj'

def readfile(filename):
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
    histogram_matrix = []
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
            i += 8

        elem = line.split()
        #print(i, timestep, elem[0], j - (num_atoms+9)*timestep)
        j = (i-(9+ 9*timestep))
        velocity_values_i[j - ((num_atoms)*timestep)] = np.linalg.norm(elem[5:])
        i +=1

    infile.close()
    return histogram_matrix


histogram_matrix= readfile(path)
norm = np.dot(histogram_matrix[0][1], histogram_matrix[0][1])
dots = np.zeros(len(histogram_matrix))
for i in range(1,len(histogram_matrix)):
    dots[i] = np.dot(histogram_matrix[0][1], histogram_matrix[i][1])/(norm)
plt.plot(dots)
plt.show()

plt.subplot(211)
plt.bar(histogram_matrix[0][1][:-1], histogram_matrix[0][0])
plt.subplot(212)
plt.bar(histogram_matrix[-1][1][:-1], histogram_matrix[-1][0])
print(histogram_matrix[-1])
print(histogram_matrix[-1][1])
print('cunt')
print(histogram_matrix[-1][0])
plt.show()
