"""
Function for reading log file and plotting the temperature. for project 1 part b
"""
import numpy as np
import matplotlib.pyplot as plt

path = 'log.lammps'

def readfile(filename):
    infile = open(filename, 'r')
    file = infile.readlines()

    i=0

    size = int(np.shape(file)[0])
    step_num = 0
    step_size = 0

    #skipping preamble, as well as extracting number of timesteps
    while i < 100: #assume preamble of log file is not more than 100 lines
        line = file[i]
        if line[:4] == "Step":
            i+=1
            print('check')
            break
        else:
            i+=1
            print(line[3:])
            if line[:3] == 'run':

                step_num = int(line[3:])
                print('cunt')
                print(step_num)
            if line[:7] == 'thermo ':
                print('twat')
                step_size = int(line[6:])

    num_lines = int(step_num/step_size) #number of lines of actual data
    print(num_lines)
    print(step_num)
    print(step_size)
    print(' ')
    step = np.zeros(num_lines+1)
    temp = np.zeros(num_lines+1)
    press = np.zeros(num_lines+1)
    kineng = np.zeros(num_lines+1)
    poteng = np.zeros(num_lines+1)
    toteng = np.zeros(num_lines+1)
    dist = np.zeros(num_lines+1)
    j = 0
    timestep = 0
    while timestep <= num_lines: #assumes log is written every 100 timesteps
        line = file[i]
        step_, temp_, kineng_, poteng_, toteng_, press_, dist_= line.split()
        step_, temp_, kineng_, poteng_, toteng_, press_, dist_ = int(step_), \
                                                                float(temp_), \
                                                                float(kineng_),\
                                                                float(poteng_),\
                                                                float(toteng_),\
                                                                float(press_),\
                                                                float(dist_)
        #print(step_, temp_, kineng_, poteng_, toteng_, press_, dist_)
        step[j] = step_
        temp[j] = temp_
        press[j] = press_
        kineng[j] = kineng_
        poteng[j] = poteng_
        toteng[j] = toteng_
        dist[j] = dist_
        print(step_, dist_)

        i +=1
        timestep += 1
        j +=1

    infile.close()
    return step, temp, press, kineng, poteng, toteng, dist




step, temp, press, kineng, poteng, toteng, dist  = histogram_matrix= readfile(path)
plt.plot(step, temp)
plt.xlabel('step')
plt.ylabel('tempterautre [Lennard Jones]')
plt.show()
plt.plot(step, dist)
plt.xlabel('step')
plt.ylabel('temp [lennar jones]')
plt.show()

plt.plot(step, kineng, label ='kineng')
plt.plot(step, poteng, label ='poteng')
plt.plot(step, toteng, label ='toteng')
plt.legend()
plt.show()
