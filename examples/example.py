import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as pl
import sys
import BEM
import pandas as pd




INPUT_PATH = pl.Path(__file__).resolve().parent.parent / "inputs"



# Read Blade Data

blade_data = BEM.Read_Blade_data(INPUT_PATH / "IEA-15-240-RWT")
# print(blade_data)

# Read and order the airfoil data
unorder_airfoil_data = BEM.Aerodynamic_file_names(INPUT_PATH / "IEA-15-240-RWT/Airfoils", "IEA-15-240-RWT_AeroDyn15_Polar_")
order_airfoil_data = BEM.Blade_order_Airfoils(blade_data, unorder_airfoil_data)
# print(order_airfoil_data)

# Group lift and drag coefficient
aerodynamic_data = BEM.Aerodynamic_inputs(order_airfoil_data, INPUT_PATH / "IEA-15-240-RWT/Airfoils")
cl_list = BEM.Aerodynamic_inputs.group_data_cl(aerodynamic_data)
# Print the cl_list as a table
cl_table = pd.DataFrame(cl_list)
# print(cl_table)

cd_list = BEM.Aerodynamic_inputs.group_data_cd(aerodynamic_data)
# Print the cl_list as a table
cd_table = pd.DataFrame(cd_list)
# print(cd_table)

#Load and order the airfoil coordinates
unorder_airfoil_coord_names = BEM.Airfoil_coord_names(INPUT_PATH / "IEA-15-240-RWT/Airfoils", "IEA-15-240-RWT_AF")
order_airfoil_coord_names = BEM.Blade_order_Airfoils(blade_data, unorder_airfoil_coord_names)

# Print blade_span_m in blade_data
# print("Blade Span (m):", blade_data['blade_span_m'])

# print(order_airfoil_coord_names)
# Plot aifoils
fig, ax = BEM.plot_airfoil (unorder_airfoil_coord_names, INPUT_PATH / "IEA-15-240-RWT/Airfoils", blade_data)

# plt.show()

test_opt_data = BEM.Blade_opt_data(INPUT_PATH / "IEA-15-240-RWT")
# print(test_opt_data)

tsr, pitch_interp, rot_speed_interp = BEM.Compute_TSR_pitch([15], test_opt_data)
print(tsr, pitch_interp, rot_speed_interp)