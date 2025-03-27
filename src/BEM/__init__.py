import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as pl


def plot_airfoil (file):
    """
    This function plots the airfoil from the given file
    """
    # Read the data from the file
    data = np.loadtxt(file, skiprows=8)
    x = data[:,0]
    y = data[:,1]
    print("Data gets stored")

    # Create the plot
    fig, axs = plt.subplots()
    axs.plot(x, y, label='Airfoil Shape')
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_title('Airfoil Plot')
    axs.legend()
    axs.axis('equal')
    axs.grid(True)

    # Export the figure and axes
    return fig, axs

