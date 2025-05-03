import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as pl
import sys
import BEM # Import the BEM module before running the script
import pandas as pd


## Inputs
#Input parameters for the wind turbine defined by IEA (International Energy Agency) 15 MW offshore reference turbine
wind_speed_start = 3
wind_speed_end = 25
wind_speed_step = 0.5
rotor_diameter = 240
rated_power = 15*1e6 # W
B = 3; # Number of blades
density = 1.225 # kg/m^3

## Paths . Find the path to the inputs folder. Inputs folder in the same directory as the script.

INPUT_PATH = pl.Path(__file__).resolve().parent.parent / "inputs"
Blade_characteristics_path = INPUT_PATH / "IEA-15-240-RWT"
Airfoil_Aerodynamic_path = INPUT_PATH / "IEA-15-240-RWT/Airfoils"
Airfoil_coord_path = INPUT_PATH / "IEA-15-240-RWT/Airfoils"
Operational_characteristics_path = INPUT_PATH / "IEA-15-240-RWT"

## File prefixes

Aifoil_Aerodynamic_file_prefix = "IEA-15-240-RWT_AeroDyn15_Polar_"
Aifoil_coord_file_prefix = "IEA-15-240-RWT_AF"

## Calculations of the rotor radius and wind speed array

rotor_radius = rotor_diameter / 2
wind_speed = np.arange(wind_speed_start, wind_speed_end + wind_speed_step, wind_speed_step)

# Read Blade Data

blade_data = BEM.Read_Blade_data(Blade_characteristics_path)

# Read and order the airfoil data using the BEM module
unorder_airfoil_data = BEM.Aerodynamic_file_names(Airfoil_Aerodynamic_path, Aifoil_Aerodynamic_file_prefix)
order_airfoil_data = BEM.Blade_order_Airfoils(blade_data, unorder_airfoil_data)


# Group lift coefficient using the Aerodynamic_inputs class from BEM module
aerodynamic_data = BEM.Aerodynamic_inputs(order_airfoil_data, Airfoil_Aerodynamic_path)
cl_list = BEM.Aerodynamic_inputs.group_data_cl(aerodynamic_data)
# cl_list as a table
cl_table = pd.DataFrame(cl_list)

# Group drag coefficient using the Aerodynamic_inputs class from BEM module
cd_list = BEM.Aerodynamic_inputs.group_data_cd(aerodynamic_data)
# cd_list as a table
cd_table = pd.DataFrame(cd_list)


#Load and order the airfoil coordinates
unorder_airfoil_coord_names = BEM.Airfoil_coord_names(Airfoil_coord_path, Aifoil_coord_file_prefix)
order_airfoil_coord_names = BEM.Blade_order_Airfoils(blade_data, unorder_airfoil_coord_names)

# Plot aifoils creates a 3-D plot of the blade by alligning all the airfoil shapes along the span
fig, ax = BEM.plot_airfoil (unorder_airfoil_coord_names, Airfoil_coord_path, blade_data)

# Load operational characteristics
test_opt_data = BEM.Blade_opt_data(Operational_characteristics_path)

# Initialize arrays for thrust, power, c_thrust and c_power
total_thrust = np.zeros(len(wind_speed))
total_power = np.zeros(len(wind_speed))
c_thrust = np.zeros(len(wind_speed))
c_power = np.zeros(len(wind_speed))
tsr = np.zeros(len(wind_speed))
pitch_interp = np.zeros(len(wind_speed))

# Calculation of Thrust, Power, CT and CP for each wind speed using the BEM module functions
for i in range(1, len(wind_speed)):
    # Compute the tip speed ratio, pitch angle and rotor speed
    tsr[i], pitch_interp[i], rot_speed_interp = BEM.Compute_TSR_pitch(wind_speed[i], test_opt_data)

    # Compute induction factors
    a, a_prime= BEM.Compute_ind_factor(wind_speed[i], rot_speed_interp, pitch_interp[i], blade_data, cl_list, cd_list, B )

    # Compute the local thrust and moment
    thrust, moment = BEM.Compute_local_thrust_moment(a, a_prime, wind_speed[i], rot_speed_interp, blade_data, density)

    # Compute the total thrust and power
    total_thrust[i], total_power[i] = BEM.Compute_Power_Thrust(thrust, moment, rot_speed_interp, rated_power, blade_data)

    # Compute the thrust and power coefficients
    c_thrust[i], c_power[i] = BEM.Compute_CT_CP(total_thrust[i], total_power[i], wind_speed[i], rotor_radius, density)

# Plot thrust, power, CT and CP using Plot_results class from BEM module
fig_thrust_power, (ax_thurst, ax1_power) = BEM.Plot_results.Plot_Power_Thrust(wind_speed, total_thrust, total_power)
fig_ct_cp, (ax_ct, ax1_cp) = BEM.Plot_results.Plot_CT_CP(wind_speed, c_thrust, c_power)
# Ploting additional functionalities: CT, CP and TSR
print("-" * 50)
print("Started processing additional functionalities")
fig_cp_ct_tsr, (ax_cp_tsr, ax_ct_tsr) = BEM.plot_CP_CT_TSR(tsr, c_power, c_thrust)
fig_power_thrust_comp, (ax2_thrust_comp, ax_power_comp) = BEM.Plot_Power_Thrust_Compare(wind_speed, total_thrust, total_power, test_opt_data)

print("Started executing the corrected model of calculation of induction factors")

total_thrust_corrected = np.zeros(len(wind_speed))
total_power_corrected = np.zeros(len(wind_speed))


# Calculation of Thrust, Power, CT and CP for each wind speed using class Corrected_ind_factors from BEM module
for i in range(1, len(wind_speed)):
    # Compute the tip speed ratio, pitch angle and rotor speed
    tsr, pitch_interp, rot_speed_interp = BEM.Compute_TSR_pitch([wind_speed[i]], test_opt_data)

    # Compute induction factors
    a_corrected, a_prime_corrected= BEM.Corrected_ind_factors.Compute_ind_factor_corrected(wind_speed[i], rot_speed_interp, pitch_interp, blade_data, cl_list, cd_list, B, rotor_radius)

    # Compute the local thrust and moment
    thrust_corrected, moment_corrected = BEM.Compute_local_thrust_moment(a_corrected, a_prime_corrected, wind_speed[i], rot_speed_interp, blade_data, density)

    # Compute the total thrust and power
    total_thrust_corrected[i], total_power_corrected[i] = BEM.Compute_Power_Thrust(thrust_corrected, moment_corrected, rot_speed_interp, rated_power, blade_data)


fig_models, (ax1_thrust_models, ax2_power_models) = BEM.Corrected_ind_factors.plot_compare_Power_Thrust_models(total_thrust, total_power, total_thrust_corrected, total_power_corrected, wind_speed)



plt.show()