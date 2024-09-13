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
    plt.scatter(fy[fell], fx[fell], c=(0.0, 0.302, 0.251), marker='x', label='Fall')  # Red color in RGB
    plt.scatter(fy[~fell], fx[~fell], c=(0, 0, 0.4), label='Success')  # Blue color in RGB
    plt.xlabel('Sideways Disturbance Force (N)')
    plt.ylabel('Forward Disturbance Force (N)')

    # Make a circle with 150N radius
    circle = plt.Circle((0, 0), 200, color='k', fill=False)
    plt.gca().add_artist(circle)
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.gca().set_aspect('equal', adjustable='box')

plt.figure(figsize=(8, 4))
plt.subplot(1,3,1)
plt.title("HLIP only")
make_scatter_plot("data_random_disturbance_hlip.csv")
plt.legend(loc='upper left')

plt.subplot(1,3,2)
plt.title("CI-MPC")
make_scatter_plot("data_random_disturbance_mpc.csv")

plt.subplot(1,3,3)
plt.title("HLIP + CI-MPC (proposed)")
make_scatter_plot("data_random_disturbance_mh.csv")

plt.tight_layout()
plt.show()
