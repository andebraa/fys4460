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
    #histogram_matrix = np.zeros((num_timesteps, num_bins, num_bins +1)) #the histogram for each timestep
    histogram_matrix = []
    timestep = 0
    i=0
    while i < int(np.shape(file)[0]):
        line = file[i]
        if line[:14] == 'ITEM: TIMESTEP':
            print('fuck')
            timestep +=1
            # hist1, hist2 = np.histogram(velocity_values_i, bins=num_bins)
            # print(np.shape(hist1))
            # print(np.shape(hist2))
            histogram_matrix.append(np.histogram(velocity_values_i, bins=num_bins))

            print(timestep)
            for i in range(8):
                i += 1#skipping number of atoms etc

        elem = file[i]
        elem = line.split()
        print(elem)
        velocity_values_i[i - num_atoms*time_step] = np.linalg.norm((float(elem[-1]), float(elem[-2]), float(elem[-3])))
        i +=1





    print(header)
    infile.close()
readfile(path)
