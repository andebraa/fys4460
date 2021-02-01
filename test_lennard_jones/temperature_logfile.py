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
    print(size)
    step_num = 0
    print(file[37][:3])
    #skipping preamble, as well as extracting number of timesteps
    while i < 100: #assume preamble of log file is not more than 100 lines
        line = file[i]
        if line[:35] != "Step Temp E_pair E_mol TotEng Press":
            i+=1
            if line[:3] == 'run':
                step_num = int(line[3:])
                print('cunt')
        elif line[:35] == "Step Temp E_pair E_mol TotEng Press":
            i+=1
            break
    print(i)
    print(step_num)
    num_lines = int(step_num/100) #number of lines of actual data
    step = np.zeros(num_lines+1)
    temp = np.zeros(num_lines+1)
    j = 0
    timestep = 0
    while timestep <= num_lines: #assumes log is written every 100 timesteps
        line = file[i]
        step_, temp_, E_pair, E_mol, TotEng, Press = line.split()
        step_, temp_, E_pair, E_mol, TotEng, Press = int(step_), float(temp_), \
                                                   float(E_pair), float(E_mol),\
                                                   float(TotEng), float(Press)
        print(step_, temp_, E_pair, E_mol, TotEng, Press)
        step[j] = step_
        temp[j] = temp_


        i +=1
        timestep += 1
        j +=1

    infile.close()
    return step, temp, E_pair, E_mol, TotEng, Press




step, temp, E_pair, E_mol, TotEng, Press = histogram_matrix= readfile(path)
plt.plot(step, temp)
plt.show()
