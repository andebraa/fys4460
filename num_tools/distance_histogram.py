import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

path = '../project1/lj_variable_temp/dump.lammps_0.5_dev'

def readfile(filename):
    """
    dumpfile reader made for calculating distance between atoms. largely based
    on 'histogram_velocity_read'
    """
    infile = open(filename, 'r')
    file = infile.readlines()
    num_atoms = int(file[3])

    #extracting header
    header = file[8].split()

    num_bins = 30 #number of bins
    num_timesteps = int(np.shape(file)[0]/(num_atoms+9))
    particle_pos = np.zeros((num_atoms, 3)) #x,y,z
    #combinations = int((num_atoms*(num_atoms-1))/2) #num_atoms pick 2, med tilbakelegging
    #distance_values = np.zeros(combinations+1)
    distance_values = []
    histogram_matrix = []


    timestep = 0
    i=9 #running index of each row of the file
    j=0 #actual particle index
    #e = 0
    size = int(np.shape(file)[0])
    while i <= (size -1):
        line = file[i]
        if line[:14] == 'ITEM: TIMESTEP':
            timestep +=1
            if timestep in (10,11,12,13,14):
                for j in range(num_atoms):
                    #thesepos stores all the positions
                    jpos = particle_pos[j]
                    for k in range(j+1, num_atoms):
                        distance_values.append(np.linalg.norm((jpos, particle_pos[k])))
                        #distance_values[e] = np.linalg.norm((jpos,particle_pos[k]))
                        #e +=1

                histogram_matrix.append(np.histogram(distance_values, bins=(np.linspace(0,20,num_bins))))
                distance_values = []
            i += 9
            line = file[i]

        elem = line.split() #ITEM: ATOMS id type x y z vx vy vz
        j = (i-(9+ 9*timestep))
        print(i, timestep, elem[5:], j - (num_atoms+9)*timestep)
        print(j - ((num_atoms)*timestep))
        particle_pos[j - ((num_atoms)*timestep)] = elem[2:5] #stores the x,y,z distance of particle j
        i +=1


    infile.close()
    return histogram_matrix

def histogram_plot():

    histogram_matrix= readfile(path)

    hist = []
    edges = []
    for i in histogram_matrix:
        hist.append(i[0])
        edges.append(i[1])
    print(np.shape(hist))
    print(np.shape(edges))
    avg_hist = np.average(hist, axis=0)
    avg_edges = np.average(edges, axis=0)
    print(np.shape(avg_hist))
    print(np.shape(avg_edges))
    print(avg_hist)
    print(avg_edges)
    plt.plot(avg_hist, avg_edges[:-1])

    plt.show()

if __name__ == '__main__':
    histogram_plot()
