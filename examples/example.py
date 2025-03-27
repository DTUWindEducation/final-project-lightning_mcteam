import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as pl
import sys
import BEM

MAIN_PATH = pl.Path(__file__).resolve().parent.parent / "inputs"

fig, ax = BEM.plot_airfoil(MAIN_PATH / "inputs/IEA-15-240-RWT/Airfoils/IEA-15-240-RWT_AF04_Coords.txt")

plt.show()