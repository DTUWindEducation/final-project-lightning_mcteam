import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as pl
import sys
import BEM
import pandas as pd


## Inputs

wind_speed_start = 3
wind_speed_end = 25
wind_speed_step = 0.5
rotor_diameter = 240
rated_power = 15*1e6 # W
B = 3; # Number of blades
density = 1.225 # kg/m^3

## Paths

INPUT_PATH = pl.Path(__file__).resolve().parent.parent / "inputs"
Blade_characteristics_path = INPUT_PATH / "IEA-15-240-RWT"
Airfoil_Aerodynamic_path = INPUT_PATH / "IEA-15-240-RWT/Airfoils"
Airfoil_coord_path = INPUT_PATH / "IEA-15-240-RWT/Airfoils"
Operational_characteristics_path = INPUT_PATH / "IEA-15-240-RWT"

## File prefixes

Aifoil_Aerodynamic_file_prefix = "IEA-15-240-RWT_AeroDyn15_Polar_"
Aifoil_coord_file_prefix = "IEA-15-240-RWT_AF"

## Calculations

rotor_radius = rotor_diameter / 2
wind_speed = np.arange(wind_speed_start, wind_speed_end + wind_speed_step, wind_speed_step)

# Read Blade Data

blade_data = BEM.Read_Blade_data(Blade_characteristics_path)
print(blade_data)

# Read and order the airfoil data
unorder_airfoil_data = BEM.Aerodynamic_file_names(Airfoil_Aerodynamic_path, Aifoil_Aerodynamic_file_prefix)
order_airfoil_data = BEM.Blade_order_Airfoils(blade_data, unorder_airfoil_data)


# Group lift coefficient
aerodynamic_data = BEM.Aerodynamic_inputs(order_airfoil_data, Airfoil_Aerodynamic_path)
cl_list = BEM.Aerodynamic_inputs.group_data_cl(aerodynamic_data)
# cl_list as a table
cl_table = pd.DataFrame(cl_list)

# Group drag coefficient
cd_list = BEM.Aerodynamic_inputs.group_data_cd(aerodynamic_data)
# cd_list as a table
cd_table = pd.DataFrame(cd_list)


#Load and order the airfoil coordinates
unorder_airfoil_coord_names = BEM.Airfoil_coord_names(Airfoil_coord_path, Aifoil_coord_file_prefix)
order_airfoil_coord_names = BEM.Blade_order_Airfoils(blade_data, unorder_airfoil_coord_names)

# Plot aifoils
fig, ax = BEM.plot_airfoil (unorder_airfoil_coord_names, Airfoil_coord_path, blade_data)

# Load operational characteristics
test_opt_data = BEM.Blade_opt_data(Operational_characteristics_path)

# Initialize arrays for thrust, power, c_thrust and c_power
total_thrust = np.zeros(len(wind_speed))
total_power = np.zeros(len(wind_speed))
c_thrust = np.zeros(len(wind_speed))
c_power = np.zeros(len(wind_speed))

# Calculation of Thrust, Power, CT and CP for each wind speed
for i in range(1, len(wind_speed)):
    # Compute the tip speed ratio, pitch angle and rotor speed
    tsr, pitch_interp, rot_speed_interp = BEM.Compute_TSR_pitch([wind_speed[i]], test_opt_data)
    
    # Compute induction factors
    a, a_prime= BEM.Compute_ind_factor(wind_speed[i], rot_speed_interp, pitch_interp, blade_data, cl_list, cd_list, B )
    
    # Compute the local thrust and moment
    thrust, moment = BEM.Compute_local_thrust_moemnt(a, a_prime, wind_speed[i], rot_speed_interp, blade_data, density)
    
    # Compute the total thrust and power
    total_thrust[i], total_power[i] = BEM.Compute_Power_Thrust(thrust, moment, rot_speed_interp, rated_power, blade_data)
    
    # Compute the thrust and power coefficients
    c_thrust[i], c_power[i] = BEM.Compute_CT_CP(total_thrust[i], total_power[i], wind_speed[i], rotor_radius, density)
    
# Plot thrust, power, CT and CP
(fig_power, ax1_power), (fig_thrust, ax_thurst) = BEM.Plot_Power_Thrust(wind_speed, total_thrust, total_power)
(fig_ct, ax_ct), (fig_cp, ax1_cp) = BEM.Plot_CT_CP(wind_speed, c_thrust, c_power)
plt.show()