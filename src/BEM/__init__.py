import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as pl
from mpl_toolkits.mplot3d import Axes3D


def Aerodynamic_file_names(file_path, common_name):
    """
    This function returns the names of the aerodynamic files in the given path
    that start with the given common_name prefix.
    """
    # Get the list of files in the directory
    files = os.listdir(file_path)
    
    # Filter for .txt files with the given prefix
    aerodynamic_files = [f for f in files if f.startswith(common_name) and f.endswith('.dat')]
    
    # Sort the files by name
    aerodynamic_files.sort()
    
    return aerodynamic_files


def Read_Blade_data(file_path, file_name = 'IEA-15-240-RWT_AeroDyn15_blade.dat'):
    """
    Reads the blade aerodynamic data from the specified file and organizes it into a structured format.

    Args:
        file_path (str): The path to the directory containing the blade data file.

    Returns:
        dict: A dictionary containing blade aerodynamic properties such as span, chord length, twist angle,
        airfoil IDs, and other related parameters.
    """

    # Define the blade data file name
    blade_file = os.path.join(file_path, file_name)

    # Read the blade data file
    with open(blade_file, 'r') as f:
        lines = f.readlines()[6:]  # Skip the first 6 lines

    # Extract the airfoil information and store additional blade data
    blade_data = {
        'blade_span_m': [],
        'curve_aero_center_m': [],
        'sweep_aero_center_m': [],
        'curve_angle_deg': [],
        'twist_angle_deg': [],
        'chord_length_m': [],
        'airfoil_id': [],
        'control_blend': [],
        'center_bend_m': [],
        'center_torsion_m': []
    }

    for line in lines:
        parts = line.split()
        # Store the blade data in the respective lists
        blade_data['blade_span_m'].append(float(parts[0]))
        blade_data['curve_aero_center_m'].append(float(parts[1]))
        blade_data['sweep_aero_center_m'].append(float(parts[2]))
        blade_data['curve_angle_deg'].append(float(parts[3]))
        blade_data['twist_angle_deg'].append(float(parts[4]))
        blade_data['chord_length_m'].append(float(parts[5]))
        blade_data['airfoil_id'].append(int(parts[6]))
        blade_data['control_blend'].append(float(parts[7]))
        blade_data['center_bend_m'].append(float(parts[8]))
        blade_data['center_torsion_m'].append(float(parts[9]))

    print("Blade aerodynamic data and additional blade properties have been organized")
    return blade_data


def Blade_order_Airfoils(blade_data, airfoil_data):
    """
    Arranges the aerodynamic file names in the order of the airfoil IDs.

    Args:
        blade_data (dict): A dictionary containing blade aerodynamic properties, including airfoil IDs.
        airfoil_data (list): A list of aerodynamic file names.

    Returns:
        list: A list of aerodynamic file names arranged in the order of airfoil IDs.
    """

    ordered_files = []
    for airfoil_id in blade_data['airfoil_id']:
        # Format the airfoil ID to match the file naming convention
        airfoil_id_str = f"{airfoil_id - 1:02d}"  # Subtract 1 to match the naming convention
        for file_name in airfoil_data:
            if airfoil_id_str in file_name:
                ordered_files.append(file_name)
                break

    return ordered_files


class Aerodynamic_inputs:
    """
    This class contains the aerodynamic inputs for the BEM model.
    """

    def __init__(self, file_list, folder_path):
        self.file_list = file_list
        self.folder_path = folder_path
        self.data = {}
        for file in file_list:
            file_path = os.path.join(folder_path, file)
            skiprows = 20 if file.endswith('03.dat') or file.endswith('02.dat') or file.endswith('01.dat') or file.endswith('00.dat') else 54
            file_data = np.loadtxt(file_path, skiprows=skiprows)
            self.data[file] = {
                'alpha': file_data[:, 0],
                'cl': file_data[:, 1],
                'cd': file_data[:, 2]
            }
        print("Data from all files gets stored")

    def group_data_cl(self):
        """
        Creates a table of lift coefficient (cl) data for each file using the alpha values of the first file as reference.

        Each column corresponds to a different values in different files, and each row corresponds to the angle of attack (alpha)

        Returns:
            dict: A dictionary with keys as file names and values as lists of cl values for each alpha.
        """
        # Use the alpha values of the first file as reference
        reference_file = self.file_list[0]
        reference_alpha = self.data[reference_file]['alpha']
        grouped_cl = {'A0A': reference_alpha.tolist()}  # Initialize with A0A column for angle of attack

        for file, file_data in self.data.items():
            grouped_cl[file] = []
            for alpha in reference_alpha:
                # Find the closest alpha value in the file data
                idx = (np.abs(file_data['alpha'] - alpha)).argmin()
                grouped_cl[file].append(file_data['cl'][idx])

        print("Grouped cl data by file using reference alpha from the first file")
        return grouped_cl

    def group_data_cd(self):
        """
        Creates a table of drag coefficient (cd) data for each file using the alpha values of the first file as reference.

        Each column corresponds to a different values in different files, and each row corresponds to the angle of attack (alpha)

        Returns:
            dict: A dictionary with keys as file names and values as lists of cd values for each alpha.
        """
        # Use the alpha values of the first file as reference
        reference_file = self.file_list[0]
        reference_alpha = self.data[reference_file]['alpha']
        grouped_cd = {'A0A': reference_alpha.tolist()}  # Initialize with A0A column for angle of attack

        for file, file_data in self.data.items():
            grouped_cd[file] = []
            for alpha in reference_alpha:
                # Find the closest alpha value in the file data
                idx = (np.abs(file_data['alpha'] - alpha)).argmin()
                grouped_cd[file].append(file_data['cd'][idx])

        print("Grouped cd data by file using reference alpha from the first file")
        return grouped_cd


def Airfoil_coord_names(file_path, common_name):
    """
    This function returns the names of the aerodynamic files in the given path
    that start with the given common_name prefix.
    """
    # Get the list of files in the directory
    files = os.listdir(file_path)
    
    # Filter for .txt files with the given prefix
    airfoil_files = [f for f in files if f.startswith(common_name) and f.endswith('.txt')]
    
    # Sort the files by name
    airfoil_files.sort()
    
    return airfoil_files


def plot_airfoil(airfoil_file_names, path, blade_data):
    """
    This function creates a 3D plot of the blade by stacking airfoil shapes along the span,
    incorporating the twist angle for each section.
    """

    span = blade_data['blade_span_m']
    twist_angles = blade_data['twist_angle_deg']

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, file in enumerate(airfoil_file_names):
        # Read the data from the file
        file_path = os.path.join(path, file)
        data = np.loadtxt(file_path, skiprows=8)
        x = np.full_like(data[:, 0], span[i])  # Assign the span value as the x-coordinate
        y = data[:, 0]
        z = data[:, 1]

        # Apply the twist angle (rotation around the x-axis)
        twist_angle_rad = np.radians(twist_angles[i])
        y_rotated = y * np.cos(twist_angle_rad) - z * np.sin(twist_angle_rad)
        z_rotated = y * np.sin(twist_angle_rad) + z * np.cos(twist_angle_rad)

        # Plot the airfoil shape at the corresponding span location
        ax.plot(x, y_rotated, z_rotated, label=f'Span {span[i]:.2f} m')

    # Set labels and title
    ax.set_xlabel('Span (x)')
    ax.set_ylabel('Airfoil x-coord (y)')
    ax.set_zlabel('Airfoil y-coord (z)')
    ax.set_title('3D Blade Plot with Twist')
    ax.grid(True)

    # Export the figure and axes
    return fig, ax


def Blade_opt_data(file_path, input_file='IEA_15MW_RWT_Onshore.opt'):
    """
    Reads the blade optimization data from the specified file and organizes it into a structured format.

    Args:
        file_path (str): The path to the directory containing the blade optimization data file.
        input_file (str): The name of the blade optimization data file.

    Returns:
        dict: A dictionary containing blade optimization properties such as span, chord length, twist angle,
        and other related parameters.               WRONG
    """
    # Define the blade optimization data file name
    opt_file = os.path.join(file_path, input_file)

    # Read the blade optimization data file
    with open(opt_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip the first 7 lines (header)

    # Extract the optimization data and store it in a structured format
    opt_data = {
        'wind speed [m/s]': [],
        'pitch [deg]': [],
        'rot. speed [rpm]': [],
        'aero power [kw]': [],
        'aero thrust [kn]': []
    }

    for line in lines:
        parts = line.split()
        # Store the optimization data in the respective lists
        opt_data['wind speed [m/s]'].append(float(parts[0]))
        opt_data['pitch [deg]'].append(float(parts[1]))
        opt_data['rot. speed [rpm]'].append(float(parts[2]))
        opt_data['aero power [kw]'].append(float(parts[3]))
        opt_data['aero thrust [kn]'].append(float(parts[4]))

    print("Blade optimization data has been successfully organized")
    return opt_data


def Compute_TSR_pitch(wind_speed, dict_opt_data, rotor_radius = 120):
    """
    Computes the Tip Speed Ratio (TSR), pitch angle, and rotational speed for a given wind speed.
    
    Args:
        wind_speed (float): The wind speed in m/s.
        blade_data (dict): A dictionary containing blade aerodynamic properties.
        dict_opt_data (dict): A dictionary containing optimization data.

    Outputs:
        tuple: A tuple containing the computed TSR, pitch angle, and rotational speed.
    """

    wind_speed_dict = dict_opt_data['wind speed [m/s]']
    pitch_dict = dict_opt_data['pitch [deg]']
    rot_speed_dict = dict_opt_data['rot. speed [rpm]']
    rot_speed_dict = [x * 2 * np.pi / 60 for x in rot_speed_dict]  # Convert to rad/s

    # Interpolate the pitch and rotational speed for the given wind speed
    pitch_interp = np.interp(wind_speed, wind_speed_dict, pitch_dict)
    rot_speed_interp = np.interp(wind_speed, wind_speed_dict, rot_speed_dict)

    tsr = rot_speed_interp * rotor_radius / wind_speed

    return tsr, pitch_interp, rot_speed_interp


def Compute_ind_factor(wind_speed, rot_speed, pitch_angle, blade_data, cl_table, cd_table, B=3):
    """
    Computes the induction factor for a given wind speed, 
    rotational speed, and pitch angle.
    """

    a = np.zeros(len(blade_data['blade_span_m']))
    a_prime = np.zeros(len(blade_data['blade_span_m']))
    a_new = np.zeros(len(blade_data['blade_span_m']))
    a_prime_new = np.zeros(len(blade_data['blade_span_m']))

    # Initial guess
    a_new[:] = 0.01
    a_prime_new[:] = 0.01

    for i in range(len(blade_data['blade_span_m'])):
        diff_a = np.abs(a_new[i] - a[i])
        diff_a_prime = np.abs(a_prime_new[i] - a_prime[i])
        iteration_count = 0
        max_iterations = 1500

        while (diff_a > 1e-4 or diff_a_prime > 1e-4) and iteration_count < max_iterations:
            c = blade_data['chord_length_m'][i]
            r = blade_data['blade_span_m'][i]
            if r == 0:
                r = 1e-6
            num = (1 - a[i]) * wind_speed
            denom = (1 + a_prime[i]) * rot_speed * r
            phi = np.arctan(num / denom)

            beta = np.radians(blade_data['twist_angle_deg'][i])
            pitch = np.radians(pitch_angle)

            sigma = B * c / (2 * np.pi * r)
            aoa = phi - (beta + pitch)

            # Interpolate cl and cd values for the given angle of attack (aoa)
            cl_key = list(cl_table.keys())[i + 1] if i + 1 < len(cl_table.keys()) else list(cl_table.keys())[-1]
            cd_key = list(cd_table.keys())[i + 1] if i + 1 < len(cd_table.keys()) else list(cd_table.keys())[-1]
            cl = np.interp(np.degrees(aoa), cl_table['A0A'], cl_table[cl_key])
            cd = np.interp(np.degrees(aoa), cd_table['A0A'], cd_table[cd_key])

            cn = cl * np.cos(phi) + cd * np.sin(phi)
            ct = cl * np.sin(phi) - cd * np.cos(phi)

            a_new[i] = 1 / ((4 * np.sin(phi) ** 2) / (sigma * cn) + 1)
            a_prime_new[i] = 1 / ((4 * np.sin(phi) * np.cos(phi)) / (sigma * ct) - 1)

            # Update values and recompute differences
            diff_a = np.abs(a_new[i] - a[i])
            diff_a_prime = np.abs(a_prime_new[i] - a_prime[i])

            a[i] = a_new[i]
            a_prime[i] = a_prime_new[i]

            iteration_count += 1
    return a, a_prime


def Compute_local_thrust_moemnt(a, a_prime, wind_speed, rot_speed, blade_data, density = 1.225):
    """
    Computes the local thrust and moment for a given wind speed, 
    rotational speed, and pitch angle.
    """

    thrust = np.zeros(len(blade_data['blade_span_m']))
    moment = np.zeros(len(blade_data['blade_span_m']))

    for i in range(len(blade_data['blade_span_m'])):
        r = blade_data['blade_span_m'][i]
        thrust[i] = 4*np.pi*r*density*(wind_speed**2)*a[i]*(1-a[i])
        moment[i] = 4*np.pi*(r**3)*density*wind_speed*rot_speed*a_prime[i]*(1-a[i])
    return thrust, moment


def Compute_Power_Thrust(thrust, moment, rot_speed, rated_power, blade_data):
    """
    Computes the total thrust and power by integrating the local thrust and moment values.

    Args:
        thrust (numpy.ndarray): Array of local thrust values for each radial segment.
        moment (numpy.ndarray): Array of local moment values for each radial segment.
        rot_speed (float): Rotational speed of the rotor in rad/s.
        rated_power (float): Rated power of the turbine in watts.
        blade_data (dict): Dictionary containing blade aerodynamic properties.

    Returns:
        tuple: A tuple containing the total thrust and total power.
    """

    r = np.asarray(blade_data['blade_span_m'])

    # Compute the total thrust by integrating over all radial segments
    total_thrust = np.trapezoid(thrust, r)

    # Compute the total power by integrating the moment over all radial segments
    total_moment = np.trapezoid(moment, r)

    total_power = total_moment * rot_speed
    if total_power > rated_power:
        total_power = rated_power

    return float(total_thrust), float(total_power)


def Compute_CT_CP(total_thrust, total_power, wind_speed, rot_radius, density = 1.225):
    """
    Computes the thrust coefficient (CT) and power coefficient (CP) based on the total thrust and power.

    Args:
        total_thrust (float): The total thrust of the rotor.
        total_power (float): The total power of the rotor.
        wind_speed (float): The wind speed at which the rotor operates.

    Returns:
        tuple: A tuple containing the thrust coefficient (CT) and power coefficient (CP).
    """

    rotor_area = np.pi * (rot_radius**2)  # Rotor area (m^2)

    # Compute CT and CP
    CT = total_thrust / (0.5 * density * wind_speed**2 * rotor_area)
    CP = total_power / (0.5 * density * wind_speed**3 * rotor_area)

    return CT, CP


def Plot_Power_Thrust(wind_speed, total_thrust, total_power):
    """
    Plots the power and thrust curves in separate plots.

    Args:
        wind_speed (numpy.ndarray): Array of wind speeds.
        total_thrust (numpy.ndarray): Array of total thrust values.
        total_power (numpy.ndarray): Array of total power values.
    """

    # Plot thrust
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('Thrust (MN)', color='tab:blue')
    ax1.plot(wind_speed, total_thrust / 1e6, color='tab:blue', label='Thrust')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    plt.title('Thrust vs Wind Speed')

    # Plot power
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Power (MW)', color='tab:red')
    ax2.plot(wind_speed, total_power/1e6, color='tab:red', label='Power')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.grid(True)
    plt.title('Power vs Wind Speed')

    return (fig1, ax1), (fig2, ax2)


def Plot_CT_CP(wind_speed, c_thrust, c_power):
    """
    Plots the thrust coefficient (CT) and power coefficient (CP) in separate plots.

    Args:
        wind_speed (numpy.ndarray): Array of wind speeds.
        c_thrust (numpy.ndarray): Array of thrust coefficients.
        c_power (numpy.ndarray): Array of power coefficients.
    """

    # Plot CT
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('CT', color='tab:blue')
    ax1.plot(wind_speed, c_thrust, color='tab:blue', label='CT')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    plt.title('CT vs Wind Speed')

    # Plot CP
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('CP', color='tab:red')
    ax2.plot(wind_speed, c_power, color='tab:red', label='CP')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.grid(True)
    plt.title('CP vs Wind Speed')

    return (fig1, ax1), (fig2, ax2)