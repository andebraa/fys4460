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
    particle_pos = np.zeros(num_atoms, 3) #x,y,z
    distance_values = np.zeros(num_atoms)
    print(num_atoms)
    #histogram_matrix = np.zeros((num_timesteps, num_bins, num_bins +1)) #the histogram for each timestep
    histogram_matrix = [] #
    timestep = 0
    i=9 #running index of each row of the file
    j=0 #actual particle index
    size = int(np.shape(file)[0])
    print(size)
    while i <= (size -1):
        line = file[i]
        if line[:14] == 'ITEM: TIMESTEP':
            timestep +=1
            histogram_matrix.append(np.histogram(velocity_values_i, bins=num_bins))
            i += 8

        elem = line.split() #ITEM: ATOMS id type x y z vx vy vz
        #print(i, timestep, elem[0], j - (num_atoms+9)*timestep)
        j = (i-(9+ 9*timestep))
        particle_pos[j - ((num_atoms)*timestep)] = elem[2:5] #stores the x,y,z distance of particle j
        velocity_values_i[j - ((num_atoms)*timestep)] = np.linalg.norm(elem[5:])
        i +=1

    for j in range(Natoms):
        #thesepos stores all the positions
        jpos = thesepos[j]
        for k in range(j+1, natoms):
            dist = jpos = thesepos[k]
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
