"""
Plots the results of the reaction kinetics example.
"""


import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import pysmc
import matplotlib.pyplot as plt
import cPickle as pickle


if __name__ == '__main__':
    with open('reaction_kinetics_particle_approximation.pickle', 'rb') as fd:
        p = pickle.load(fd)
    print p.num_particles
    plt.plot(p.k1, p.k2, '.')
    plt.show()
