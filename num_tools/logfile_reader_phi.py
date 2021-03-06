"""
Function for reading log file and plotting the temperature. for project 1 part b
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
import os

#path = '../project2/pores/log.dev_phi0.14845113982142'
#path = '../project2/pores/data/'
class logfile_reader():
    """
    class for reading and extracting logfiles
    """
    def __init__(self, path, phi = False):
        """
        path:
            relative path to dump file

        NOTE current naming convention:
        log.<something>_T<t_value>_phi<phi_value>
        """
        self.base_path = path
        self.phi = phi


        #for PHI variations
        if phi:
            self.filenames = os.listdir(path)
            logfiles = [i for i in self.filenames if 'log' in i] #gather up the logfiles
            files = {}

            for i in logfiles:
                print(i)
                elem = re.split(r'_phi|T',i)#.strip('_') #NOTE FOLLOW NAMING CONVENSION
                print(elem)
                #files = {i:[elem[0], elem[2]]} #dictionary, {filename:[phi,T]}
                files.update({i:[elem[1], elem[2]]})
                print('cunt')

            self.files = files
            print(files)


    def readfile(self, filename = ''):
        """
        function for reading out logfiles generated by lammps. NOTE assumes the first
        Function takes one temperature. Loops are done is the respective functions

        args:
            filename (str): filename/path to the log file
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


        try:
            infile = open(filename, 'r')
        except:
            infile = open(self.base_path + filename, 'r')
        file = infile.readlines()

        i=0

        size = int(np.shape(file)[0])
        step_num = 0
        step_size = 10 #thermo 10 call

        #skipping preamble, as well as extracting number of timesteps
        while i < 100: #assume preamble of log file is not more than 100 lines
            line = file[i]
            if line[:4] == "Step":
                i+=1
                args = line.split()
                if len(args) != 8: #checking if maybe some collumns are missing or different
                    raise ValueError('Number of collumns in logfile does not match with expected')
                break
            else:
                i+=1
                if line[:3] == 'run':
                    step_num = int(line[3:])

                if line[:7] == 'thermo ':
                    step_size = int(line[6:])


        num_lines = int(step_num/step_size) #number of lines of actual data
        step   =    np.zeros(num_lines+1)
        time   =    np.zeros(num_lines+1)
        temp   =    np.zeros(num_lines+1)
        press  =    np.zeros(num_lines+1)
        kineng =    np.zeros(num_lines+1)
        poteng =    np.zeros(num_lines+1)
        toteng =    np.zeros(num_lines+1)
        dist   =    np.zeros(num_lines+1)
        #cm_vel =    np.zeros(num_lines+1)
        j = 0
        timestep = 0
        while timestep <= num_lines: #assumes log is written every 100 timesteps
            line = file[i]
            step_, time_, temp_, kineng_, poteng_, toteng_, press_, dist_= line.split()
            step_, time_, temp_, kineng_, poteng_, toteng_, press_, dist_ = \
                                                                    int(step_),    \
                                                                    float(time_),    \
                                                                    float(temp_),  \
                                                                    float(kineng_),\
                                                                    float(poteng_),\
                                                                    float(toteng_),\
                                                                    float(press_), \
                                                                    float(dist_),  #\
                                                                    #float(cm_vel_)

            step[j] = step_
            time[j] = time_
            temp[j] = temp_
            press[j] = press_
            kineng[j] = kineng_
            poteng[j] = poteng_
            toteng[j] = toteng_
            dist[j] = dist_
            #cm_vel[j] = cm_vel_

            i +=1
            timestep += 1
            j +=1

        infile.close()
        self.step = step
        self.time = time
        self.temp = temp
        self.press = press
        self.kineng = kineng
        self.poteng = poteng
        self.toteng = toteng
        self.dist = dist
        #self.cm_vel = cm_vel
        return step, time, temp, press, kineng, poteng, toteng, dist #, cm_vel






    def temp_plot(self, temps=''):
        """
        Plots tempterature as a function of time (steps)

        args:
            temps (list): List of ints. must be list!

        returns:
            nothing. Only plots
        """

        # #c part 1 temp varying init v
        # if isinstance(temps, list):
        #     for i,t in enumerate(temps):
        #         step, time, temp, press, kineng, poteng, toteng, dist, cm_vel  = self.readfile(t)
        #         avg_temp = np.average(temp)
        #         plt.plot(step, temp, label=f'avg: {temp[0]}')
        #     plt.xlabel('step')
        #     plt.ylabel('tempterautre [Lennard Jones]')


        #else:
        try:
            temp = self.temp
            step = self.step
        except:
            step, time, temp, press, kineng, poteng, toteng, dist, cm_vel  = self.readfile(v)

        plt.plot(step, temp, label=f't0 = {temp[0]}')
        plt.xlabel('step')
        plt.ylabel('tempterautre [Lennard Jones]')



    def press_plot(self, temps =''):
        """
        Plots tempterature as a function of time (steps)

        args:
            temps (list): List of ints. must be list!

        returns:
            nothing. Only plots
        """

        # #c part 1 temp varying init v
        # if isinstance(temps, list):
        #     for i,t in enumerate(temps):
        #         step, time, temp, press, kineng, poteng, toteng, dist, cm_vel  = self.readfile(t)
        #         avg_temp = np.average(temp)
        #         plt.plot(step, temp, label=f'avg: {temp[0]}')
        #     plt.xlabel('step')
        #     plt.ylabel('tempterautre [Lennard Jones]')
        #
        #
        # else:
        try:
            press = self.temp
            step = self.step
        except:
            step, time, temp, press, kineng, poteng, toteng, dist, cm_vel  = self.readfile()

        try:
            plt.plot(step, press, label=f't0 = {temp[0]}')
        except: #is temps is not a list
            plt.plot(step, press)
        plt.xlabel('step')
        plt.ylabel('pressure [Lennard Jones]')


    def dist_plot(self, temp=''):


        files = [i for i in self.files] #iterate over dictionary keys
        #oppgave f)
        for file in self.files:

            step, time, temp, press, kineng, poteng, toteng, dist = self.readfile(file)

            plt.plot(step, dist, label=f'phi:{self.files[file][1]} T:{self.files[file][0]}')
            plt.xlabel('time')
            plt.ylabel('dist [lj]')

            #part f

            pol_fit = np.polyfit(step[-100:], dist[-100:], 1)
            print(pol_fit)
            def polfit(x):
                return pol_fit[0]*x + pol_fit[1]

            x = np.linspace(step[0], step[-1], 100)
            plt.plot(x, polfit(x), '--', label=f'a={pol_fit[0]:.6f}, b={pol_fit[1]:.6f}')
            print(pol_fit[0]/(6*time[-1]))


    def average_press_plot(self, temps = ''):

        avg_press = np.zeros(len(temps))
        for i,v in enumerate(temps):
            step, time, temp, press, kineng, poteng, toteng, dist, cm_vel  = self.readfile(v)
            len = len(step)
            avg_press[i] = np.average(press)

        #oppgave d)
        plt.plot(temps, avg_press, 'o')
        plt.title('Average pressure over time,  per temperature')
        plt.xlabel('T [LJ]')
        plt.ylabel('average pressure')

    def energy_plot(self, temps=['']):
        try:
            step = self.step
            kineng = self.kineng
            poteng = self.poteng
            toteng = self.toteng
        except:
            pass

        #part B, plot energy
        plt.plot(step, kineng, label ='kineng')
        plt.plot(step, poteng, label ='poteng')
        plt.plot(step, toteng, label ='toteng')
        plt.xlabel('timestep /10')
        plt.ylabel('Energy [JL]')

    def permeability(self, temps=['']):
        """
        measures permeability as a functin of phi. requires multiple files
        """
        if temps==['']:
            raise ValueError('requires multiple temperatures')
            sys.exit(1)
        phi = np.zeros(len(temps))
        for i,v in enumerate(temps):
            step, time, temp, press, kineng, poteng, toteng, dist, cm_vel  = self.readfile(temps)






if __name__ == '__main__':
    #temps = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0']
    #temps_int = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    temps = ''
    init = logfile_reader('../project2/pores/data/', phi=True)
    #init.press_plot(temps_int)
    init.dist_plot()
    #WATER
    # energy_plot_init = logfile_reader('../project1/water/')
    # _ = energy_plot_init.readfile('log.lammps_water_100')
    # energy_plot_init.dist_plot(100)
    # _ = energy_plot_init.readfile('log.lammps_water_1000')


    plt.legend()
    plt.show()
