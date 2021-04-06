import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

#path = 'dump.lammpstrj'
path = '../project1/lj_variable_temp/dump.lammps_0.5_dev'
class dumpfile_reader():
    """
    class for reading and extracting dumpfile items.
    Be careful og dumpfile format.
    """

    def __init__(self, path, variable_temps = None):
        """
        path:
            relative path to dump file

        variable_temps (list or array of floats/int):
            If many dumpfiles with the temp/velocity as file ending
            exists this will accomodate it. It assumes names such as
            dump.lammps_0, dump.lammps_1 etc etc
        """
        self.path = path
        if variable_temps:
            self.temps = [str(i) for i in a]
        else:
            self.temps = ['']

    def readfile(self, timestep_eval = (10,11,12) ,dist_eval = False):
        filename = self.path

        infile = open(filename, 'r')
        file = infile.readlines()
        num_atoms = int(file[3])

        #extracting header
        header = file[8].split()

        num_bins = 30 #number of bins
        num_timesteps = int(np.shape(file)[0]/(num_atoms+9))
        #dinstance histograms
        particle_pos = np.zeros((num_atoms, 3)) #x,y,z
        #combinations = int((num_atoms*(num_atoms-1))/2) #num_atoms pick 2, med tilbakelegging

        velocity_values_i = np.zeros(num_atoms)
        #histogram_matrix = np.zeros((num_timesteps, num_bins, num_bins +1)) #the histogram for each timestep
        histogram_matrix = []

        #distance_values = np.zeros(combinations+1)
        distance_values = []

        timestep = 0
        i=9
        j=0
        size = int(np.shape(file)[0])
        while i <= (size -1):
            line = file[i]
            if line[:14] == 'ITEM: TIMESTEP':
                timestep +=1

                if dist_eval and timestep in timestep_eval: #for extracting inter-particle distances
                    for j in range(num_atoms):
                        #thesepos stores all the positions
                        jpos = particle_pos[j]
                        for k in range(j+1, num_atoms):
                            distance_values.append(np.linalg.norm((jpos, particle_pos[k])))
                            #distance_values[e] = np.linalg.norm((jpos,particle_pos[k]))
                            #e +=1

                    histogram_matrix.append(np.histogram(distance_values, bins=(np.linspace(0,20,num_bins))))
                    distance_values = []
                elif dist_eval == False:
                    histogram_matrix.append(np.histogram(velocity_values_i, bins=num_bins))
                i += 9
                line = file[i]

            elem = line.split()
            j = (i-(9+ 9*timestep))
            #print(i, timestep, elem[5:], j - (num_atoms+9)*timestep)
            #print(j - ((num_atoms)*timestep))
            velocity_values_i[j - ((num_atoms)*timestep)] = np.linalg.norm(elem[5:])
            particle_pos[j - ((num_atoms)*timestep)] = elem[2:5] #stores the x,y,z distance of particle j
            i +=1

        infile.close()
        self.histogram_matrix = histogram_matrix
        return histogram_matrix


    def histogram_evolution(self):
        """
        computes the normalized dot products of each histogram, as it evelops over time

        """
        histogram_matrix = self.readfile(path)
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
        plt.show()

    def distance_histogram(self):
        """
        For finding the distances between each particle, and plotting the histogram
        of theese. Calls readfile with dist_eval = True.
        Also doesn't work yet
        """
        histogram_matrix= self.readfile(dist_eval = True)

        hist = []
        edges = []
        print(np.shape(histogram_matrix))

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
        plt.bar(avg_hist, avg_edges[:-1])

        plt.show()

if __name__ == '__main__':
    init = dumpfile_reader(path)
    init.distance_histogram()
    #init.histogram_evolution()
