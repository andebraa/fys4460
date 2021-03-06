"""
Function for reading log file and plotting the temperature. for project 1 part b
"""
import numpy as np
import matplotlib.pyplot as plt

path = 'log.lammps_'

def readfile(filename, velocity):
    """
    function for reading out logfiles generated by lammps. NOTE assumes the first
    writeout is STEP

    args:
        filename (str): filename/path to the log file
        velocity (float): (temp) the velocity of the given simulation
    returns:
        step (ndarray): step number
        temp (ndarray): temperature for given step number
        press (ndarray): pressure for given step number
        kineng (ndarray): kinetic energy for given time step
        poteng (ndarray): potential energy for given time step
        toteng (ndarray): total energy for given time step
        dist (ndarray): average distance travelled by all the atoms computed by
                        lammps
    """
    print(filename)
    infile = open(filename+velocity, 'r')
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
                print(step_num)
            if line[:7] == 'thermo ':
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

        step[j] = step_
        temp[j] = temp_
        press[j] = press_
        kineng[j] = kineng_
        poteng[j] = poteng_
        toteng[j] = toteng_
        dist[j] = dist_
        print(step_)

        i +=1
        timestep += 1
        j +=1

    infile.close()
    return step, temp, press, kineng, poteng, toteng, dist



temps = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0']
temps_int = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# step, temp, press, kineng, poteng, toteng, dist = np.zeros(len(temps)),\
#                                                  np.zeros(len(temps)),\
#                                                  np.zeros(len(temps)),\
#                                                  np.zeros(len(temps)),\
#                                                  np.zeros(len(temps)),\
#                                                  np.zeros(len(temps)),\
#                                                  np.zeros(len(temps))
avg_press = np.zeros(len(temps))
for i,v in enumerate(temps):
    step, temp, press, kineng, poteng, toteng, dist  = histogram_matrix= readfile(path, v)

    # #c part 1 temp varying init v
    # plt.plot(step, temp, label=f'v = {v}')
    # plt.xlabel('step')
    # plt.ylabel('tempterautre [Lennard Jones]')

    # #c part 2 pressure varying init v
    # plt.plot(step, press, label = f'v = {v}')
    # plt.xlabel('step')
    # plt.ylabel('pressure [LJ]')

    #oppgave f)
    plt.plot(step, dist, label=f'v:{v}')
    plt.xlabel('step')
    plt.ylabel('dist [lj]')


    #oppgave d)
    avg_press[i] = np.average(press)

#oppgave d)
# plt.plot(temps, avg_press, 'o')
# plt.title('Average pressure over time,  per temperature')
# plt.xlabel('T [LJ]')
# plt.ylabel('average pressure')


#part B, plot energy
# plt.plot(step, kineng, label ='kineng')
# plt.plot(step, poteng, label ='poteng')
# plt.plot(step, toteng, label ='toteng')
# plt.xlabel('timestep /10')
# plt.ylabel('Energy [JL]')


#part f
pol_fit = np.polyfit(step[-100:], dist[-100:], 1)
print(pol_fit)
def polfit(x):
    return pol_fit[0]*x + pol_fit[1]

x = np.linspace(step[0], step[-1], 1000)
plt.plot(x, polfit(x) )

plt.legend()
plt.show()
