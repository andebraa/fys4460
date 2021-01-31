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
    while i <= size:
        line = file[i]
        if line[:14] == 'ITEM: TIMESTEP':
            print('fuck')
            timestep +=1
            histogram_matrix.append(np.histogram(velocity_values_i, bins=num_bins))

            print(timestep)
            i += 8

        elem = line.split()
        print(i, timestep, elem[0], j - (num_atoms+9)*timestep)
        j = (i-9*timestep)
        velocity_values_i[j - ((num_atoms-9)*timestep)] = np.linalg.norm(elem[5:])
        i +=1



    print(header)
    infile.close()
readfile(path)
