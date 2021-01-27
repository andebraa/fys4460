import numpy as np
import matplotlib.pyplot as plt

path = 'dump.lammpstrj'

def readfile(filename):
    infile = open(filename, 'r')
    x = infile.readline()
    while x[:11] != 'ITEM: ATOMS':
        x= infile.readline()
    x = x.split()
    header = x[2:]
    print(header)

    # for line in infile:
        # elem = line.split()
    infile.close()
readfile(path)
