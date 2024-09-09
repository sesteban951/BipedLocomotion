#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def make_scatter_plot(fname):
    """Make a scatter plot of forces and whether the robot fell or not."""
    # Load data from the csv file
    data = np.genfromtxt(fname, delimiter=',', skip_header=1)

    # Parse the data
    fx = data[:, 0]
    fy = data[:, 1]
    fell = data[:, 3] > 0

    # Make a scatter plot of the data
    plt.scatter(fx[~fell], fy[~fell], c='b', label='Success')
    plt.scatter(fx[fell], fy[fell], c='r', marker='x', label='Fall')
    plt.xlabel('Forward Disturbance Force (N)')
    plt.ylabel('Sideways Disturbance Force (N)')

    # Make a circle with 150N radius
    circle = plt.Circle((0, 0), 150, color='k', fill=False)
    plt.gca().add_artist(circle)
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.gca().set_aspect('equal', adjustable='box')

plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
plt.title("HLIP only")
make_scatter_plot("data_random_disturbance_hlip.csv")
plt.legend(loc='upper left')

plt.subplot(1,2,2)
plt.title("HLIP + CI-MPC (proposed)")
make_scatter_plot("data_random_disturbance_mpc.csv")
plt.tight_layout()
plt.show()