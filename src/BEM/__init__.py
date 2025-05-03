"""
This module provides a comprehensive set of functions and classes for analyzing
and modeling the aerodynamic performance of wind turbine blades using the
Blade Element Momentum (BEM) theory. It includes utilities for reading and
processing blade data, computing aerodynamic coefficients, and visualizing
results.
"""


import numpy as np
import matplotlib.pyplot as plt
import os


def Aerodynamic_file_names(file_path, common_name):
    """
    Retrieves a sorted list of aerodynamic file names from a
    specified directory. This function scans the specified
    directory for files that start with a given prefix
    (`common_name`) and have a `.dat` extension. It returns a
    list of such file names sorted in ascending order.

    Args:
        file_path (str): The path to the directory containing
        the files.
        common_name (str): The prefix to filter the file names.

    Returns:
        list of str: A sorted list of aerodynamic file names
        matching the criteria.

    Raises:
        FileNotFoundError: If the specified directory does not
        exist.
        PermissionError: If there are insufficient permissions
        to access the directory.
    """
    # Get the list of files in the directory
    files = os.listdir(file_path)
    # Filter files that start with specified common name and end with '.dat'
    aerodynamic_files = [
        f for f in files if f.startswith(common_name) and f.endswith('.dat')
    ]
    # Sort the filtered list of aerodynamic files in ascending order
    aerodynamic_files.sort()

    return aerodynamic_files


def Read_Blade_data(file_path, file_name='IEA-15-240-RWT_AeroDyn15_blade.dat'):
    """
    Reads and processes blade aerodynamic data and additional
    blade properties from a specified data file.
    Args:
        file_path (str): The directory path where the blade
        data file is located.
        file_name (str, optional): The name of the blade data
        file. Defaults to 'IEA-15-240-RWT_AeroDyn15_blade.dat'.
    Returns:
        dict: A dictionary containing the following keys and
        their corresponding lists of values:
            - 'blade_span_m': List of blade span positions in
              meters.
            - 'curve_aero_center_m': List of curve aerodynamic
              center positions in meters.
            - 'sweep_aero_center_m': List of sweep aerodynamic
              center positions in meters.
            - 'curve_angle_deg': List of curve angles in degrees.
            - 'twist_angle_deg': List of twist angles in degrees.
            - 'chord_length_m': List of chord lengths in meters.
            - 'airfoil_id': List of airfoil IDs as integers.
            - 'control_blend': List of control blend values.
            - 'center_bend_m': List of center bend positions in
              meters.
            - 'center_torsion_m': List of center torsion
              positions in meters.
    Notes:
        - The function skips header lines and the first 6 lines
          of the file.
        - Lines starting with "#" or empty lines are ignored.
        - The data is expected to be space-separated in the file.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the data in the file is not in the
        expected format.
    """

    # Define the blade data file name
    blade_file = os.path.join(file_path, file_name)

    # Read the blade data file
    with open(blade_file, 'r') as f:
        lines = [
            line.strip() for line in f.readlines()
            if line.strip() and not line.strip().startswith("#")
        ][6:]  # Skip header lines and first 6 lines

    # Extract the airfoil information and store additional blade data
    blade_data = {
        'blade_span_m': [], 'curve_aero_center_m': [],
        'sweep_aero_center_m': [], 'curve_angle_deg': [],
        'twist_angle_deg': [], 'chord_length_m': [],
        'airfoil_id': [], 'control_blend': [],
        'center_bend_m': [], 'center_torsion_m': []
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

    print("Blade aerodynamic data and additional blade properties organized")
    return blade_data


def Blade_order_Airfoils(blade_data, airfoil_data):
    """
    Arranges aerodynamic file names by airfoil IDs.

    Args:
        blade_data (dict): Blade data with airfoil IDs.
        airfoil_data (list): List of aerodynamic file names.

    Returns:
        list: File names ordered by airfoil IDs.
    """
    # Convert airfoil IDs to strings with leading zeros
    ordered_files = []
    # Iterate through the airfoil IDs in the blade data
    for airfoil_id in blade_data['airfoil_id']:
        airfoil_id_str = f"{airfoil_id - 1:02d}"
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
            skiprows = (
                20 if file.endswith(('03.dat', '02.dat', '01.dat', '00.dat'))
                else 54
            )
            file_data = np.loadtxt(file_path, skiprows=skiprows)
            self.data[file] = {
                'alpha': file_data[:, 0],
                'cl': file_data[:, 1],
                'cd': file_data[:, 2]
            }
        print("Data from all files gets stored")

    def group_data_cl(self):
        """
        Groups lift coefficient (cl) data from multiple files
        based on the angle of attack (alpha) values from the
        first file in the file list. This function uses the
        alpha values from the first file as a reference and
        finds the closest corresponding cl values for each
        alpha in the other files. The grouped data is returned
        as a dictionary where the keys are the file names and
        the values are lists of cl values corresponding to the
        reference alpha values.
        Returns:
            dict: A dictionary containing grouped cl data.
            The keys are:
            - 'A0A': A list of reference alpha values from
              the first file.
            - File names: Lists of cl values corresponding
              to the reference alpha values.
        """

        reference_file = self.file_list[0]
        reference_alpha = self.data[reference_file]['alpha']
        grouped_cl = {'A0A': reference_alpha.tolist()}

        # Iterate through each file and its corresponding data
        for file, file_data in self.data.items():
            grouped_cl[file] = []  # Initialize an empty list for current file
            for alpha in reference_alpha:
                # Find the index of the closest alpha value in the file data
                idx = (np.abs(file_data['alpha'] - alpha)).argmin()
                # Append the corresponding cl value to the grouped data
                grouped_cl[file].append(file_data['cl'][idx])

        print("Grouped cl data by file using reference aoa from the 1st file")
        return grouped_cl

    def group_data_cd(self):
        """
        Groups drag coefficient (cd) data from multiple files
        based on the angle of attack (alpha) values of the first
        file in the file list. Uses the alpha values from the
        first file as a reference and finds the closest matching
        alpha values in the other files to group their cd values.

        Returns:
            dict: A dictionary where:
            - 'A0A': Reference alpha values as a list.
            - File names: Lists of cd values matched to the
              reference alpha values.

        Notes:
            - Assumes `self.file_list` is a list of file names.
            - Assumes `self.data` maps file names to dictionaries
              containing 'alpha' and 'cd' arrays.
            - Closest alpha value is determined using minimum
              absolute difference.

        Prints:
            A message indicating that cd data has been grouped
            using reference alpha values from the first file.
        """

        # Use the alpha values of the first file as reference
        reference_file = self.file_list[0]
        reference_alpha = self.data[reference_file]['alpha']
        grouped_cd = {'A0A': reference_alpha.tolist()}

        # Iterate through each file and its corresponding data
        for file, file_data in self.data.items():
            grouped_cd[file] = []  # Initialize an empty list for current file
            for alpha in reference_alpha:
                # Find the index of the closest alpha value in the file data
                idx = (np.abs(file_data['alpha'] - alpha)).argmin()
                # Append the corresponding cd value to the grouped data
                grouped_cd[file].append(file_data['cd'][idx])

        print("Grouped cd data by file using reference aoa from the 1st file")
        return grouped_cd


def Airfoil_coord_names(file_path, common_name):
    """
    Retrieves a sorted list of airfoil coordinate file names
    from a specified directory. This function scans the given
    directory for files that start with the specified common
    name and end with the '.txt' extension. It then returns a
    sorted list of these file names.
    Args:
        file_path (str): The path to the directory containing
        the airfoil files.
        common_name (str): The common prefix of the airfoil
        file names to search for.
    Returns:
        list: A sorted list of file names matching the
        specified criteria.
    """

    # Get the list of files in the directory
    files = os.listdir(file_path)
    # Filter files that start with specified common name and end with '.txt'
    airfoil_files = [
        f for f in files if f.startswith(common_name) and f.endswith('.txt')
    ]
    # Sort the filtered list of airfoil files in ascending order
    airfoil_files.sort()

    return airfoil_files


def plot_airfoil(airfoil_file_names, path, blade_data):
    """
    Plots a 3D representation of airfoil shapes along the span
    of a blade, incorporating twist angles at each span location.
    Parameters:
    -----------
    airfoil_file_names : list of str
        A list of filenames containing airfoil coordinate data.
        Each file should contain airfoil coordinates with the
        first column as x-coordinates and the second column as
        y-coordinates.
    path : str
        The directory path where the airfoil files are located.
    blade_data : dict
        A dictionary containing blade-related data with the
        following keys:
        - 'blade_span_m': list of float
            The spanwise positions (in meters) along the blade
            where the airfoils are located.
        - 'twist_angle_deg': list of float
            The twist angles (in degrees) at each spanwise
            position.
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the 3D plot.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The matplotlib 3D axes object for the plot.
    Notes:
    ------
    - The airfoil files are expected to have a header of at
      least 8 rows, which will be skipped when reading the data.
    - The twist angle is applied as a rotation around the x-axis.
    - The function labels each airfoil plot with its
      corresponding span location.
    Example:
    --------
    >>> airfoil_files = ['airfoil1.dat', 'airfoil2.dat',
    ...                  'airfoil3.dat']
    >>> path = '/path/to/airfoil/files'
    >>> blade_data = {
    ...     'blade_span_m': [0.0, 5.0, 10.0],
    ...     'twist_angle_deg': [0.0, 5.0, 10.0]
    ... }
    >>> fig, ax = plot_airfoil(airfoil_files, path, blade_data)
    >>> plt.show()
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
        x = np.full_like(data[:, 0], span[i])
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
    Reads the blade optimization data from the specified file.

    Args:
        file_path (str): Path to the directory containing the file.
        input_file (str): Name of the blade optimization data file.

    Returns:
        dict: A dictionary containing optimization data with keys:
            - 'wind speed [m/s]': List of wind speeds.
            - 'pitch [deg]': List of pitch angles.
            - 'rot. speed [rpm]': List of rotational speeds.
            - 'aero power [kw]': List of aerodynamic power values.
            - 'aero thrust [kn]': List of aerodynamic thrust values.
    """
    # Define the blade optimization data file name
    opt_file = os.path.join(file_path, input_file)

    # Read the blade optimization data file
    with open(opt_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip the header line

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


def Compute_TSR_pitch(wind_speed, dict_opt_data, rotor_radius=120):
    """
    Computes the Tip Speed Ratio (TSR), pitch angle, and
    rotational speed for a given wind speed.

    Args:
        wind_speed (float): The wind speed in m/s.
        blade_data (dict): A dictionary containing blade
        aerodynamic properties.
        dict_opt_data (dict): A dictionary containing
        optimization data.

    Outputs:
        tuple: A tuple containing the computed TSR, pitch
        angle, and rotational speed.
    """

    wind_speed_dict = dict_opt_data['wind speed [m/s]']
    pitch_dict = dict_opt_data['pitch [deg]']
    rot_speed_dict_rpm = dict_opt_data['rot. speed [rpm]']
    # Convert to rad/s
    rot_speed_dict = [x * 2 * np.pi / 60 for x in rot_speed_dict_rpm]

    # Interpolate the pitch and rotational speed for the given wind speed
    pitch_interp = np.interp(wind_speed, wind_speed_dict, pitch_dict)
    rot_speed_interp = np.interp(wind_speed, wind_speed_dict, rot_speed_dict)

    tsr = rot_speed_interp * rotor_radius / wind_speed

    return tsr, pitch_interp, rot_speed_interp


def Compute_ind_factor(wind_speed, rot_speed, pitch_angle, blade_data,
                       cl_table, cd_table, B=3):

    """
    Computes the induction factor for a given wind speed,
    rotational speed, and pitch angle.

    Args:
        wind_speed (float): The wind speed in m/s.
        rot_speed (float): The rotational speed in rad/s.
        pitch_angle (float): The pitch angle in degrees.
        blade_data (dict): A dictionary containing blade aerodynamic
        properties.
        cl_table (dict): A dictionary containing lift coefficient data.
        cd_table (dict): A dictionary containing drag coefficient data.
        B (int, optional): Number of blades. Defaults to 3.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - a: Axial induction factors for each blade section.
            - a_prime: Tangential induction factors for each blade section.
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

        while (diff_a > 1e-4 or diff_a_prime > 1e-4) and \
                iteration_count < max_iterations:
            c = blade_data['chord_length_m'][i]
            r = blade_data['blade_span_m'][i]
            if r == 0:
                r = 1e-6
            num = (1 - a[i]) * wind_speed
            denom = (1 + a_prime[i]) * rot_speed * r
            epsilon = 1e-6  # Small value to prevent division by zero
            phi = np.arctan(num / (denom + epsilon))

            beta = np.radians(blade_data['twist_angle_deg'][i])
            pitch = np.radians(pitch_angle)

            sigma = B * c / (2 * np.pi * r)
            aoa = phi - (beta + pitch)

            # Interpolate cl and cd values for the given aoa
            cl_key = (list(cl_table.keys())[i + 1]
                      if i + 1 < len(cl_table.keys())
                      else list(cl_table.keys())[-1])
            cd_key = (list(cd_table.keys())[i + 1]
                      if i + 1 < len(cd_table.keys())
                      else list(cd_table.keys())[-1])
            cl = np.interp(np.degrees(aoa), cl_table['A0A'],
                           cl_table[cl_key])
            cd = np.interp(np.degrees(aoa), cd_table['A0A'],
                           cd_table[cd_key])

            cn = cl * np.cos(phi) + cd * np.sin(phi)
            ct = cl * np.sin(phi) - cd * np.cos(phi)

            a_new[i] = 1 / ((4 * np.sin(phi) ** 2) / (sigma * cn) + 1)
            a_prime_new[i] = 1 / ((4 * np.sin(phi) * np.cos(phi)) /
                                  (sigma * ct) - 1)

            # Update values and recompute differences
            diff_a = np.abs(a_new[i] - a[i])
            diff_a_prime = np.abs(a_prime_new[i] - a_prime[i])

            a[i] = a_new[i]
            a_prime[i] = a_prime_new[i]

            iteration_count += 1
    return a, a_prime


def Compute_local_thrust_moment(a, a_prime, wind_speed, rot_speed, blade_data,
                                density=1.225):
    """
    Compute the local thrust and moment along the blade span.
    Parameters:
    -----------
    a : array-like
        Axial induction factors at each blade span location.
    a_prime : array-like
        Tangential induction factors at each blade span location.
    wind_speed : float
        Free-stream wind speed in meters per second (m/s).
    rot_speed : float
        Rotational speed of the turbine in radians per second.
    blade_data : dict
        Dictionary containing blade geometry data. Must include
        the key 'blade_span_m', which is an array-like object
        representing the radial positions (in meters) along the
        blade span.
    density : float, optional
        Air density in kilograms per cubic meter (kg/m³). Default
        is 1.225 (standard air density at sea level).
    Returns:
    --------
    thrust : numpy.ndarray
        Array of thrust forces (in Newtons) at each blade span.
    moment : numpy.ndarray
        Array of moments (in Newton-meters) at each blade span.
    Notes:
    ------
    - The thrust and moment are computed using the Blade Element
      Momentum (BEM) theory.
    - Ensure that the input arrays `a`, `a_prime`, and
      `blade_data['blade_span_m']` have the same length.
    """
    thrust = np.zeros(len(blade_data['blade_span_m']))
    moment = np.zeros(len(blade_data['blade_span_m']))

    for i in range(len(blade_data['blade_span_m'])):
        r = blade_data['blade_span_m'][i]
        thrust[i] = 4*np.pi*r*density*(wind_speed**2)*a[i]*(1-a[i])
        moment[i] = (4 * np.pi * r**3 * density * wind_speed *
                     rot_speed * a_prime[i] * (1 - a[i]))
    return thrust, moment


def Compute_Power_Thrust(thrust, moment, rot_speed, rated_power, blade_data):
    """
    Computes the total thrust and power by integrating the
    local thrust and moment values.

    Args:
        thrust (numpy.ndarray): Array of local thrust values
        for each radial segment.
        moment (numpy.ndarray): Array of local moment values
        for each radial segment.
        rot_speed (float): Rotational speed of the rotor in
        rad/s.
        rated_power (float): Rated power of the turbine in
        watts.
        blade_data (dict): Dictionary containing blade
        aerodynamic properties.

    Returns:
        tuple: A tuple containing the total thrust and total
        power.
    """

    r = np.asarray(blade_data['blade_span_m'])

    # Compute the total thrust by integrating over all radial segments
    total_thrust = np.trapz(thrust, r)

    # Compute the total power by integrating moment over all radial segments
    total_moment = np.trapz(moment, r)

    total_power = total_moment * rot_speed

    # Limit the total power to the rated power if it exceeds the rated power
    if total_power > rated_power:
        total_power = rated_power

    return float(total_thrust), float(total_power)


def Compute_CT_CP(total_thrust, total_power, wind_speed, rot_radius,
                  density=1.225):
    """
    Computes the thrust coefficient (CT) and power coefficient (CP) based on
    the total thrust and power.

    Args:
        total_thrust (float): The total thrust of the rotor.
        total_power (float): The total power of the rotor.
        wind_speed (float): The wind speed at which the rotor operates.

    Returns:
        tuple: A tuple containing the thrust coefficient (CT) and power
        coefficient (CP).
    """

    rotor_area = np.pi * (rot_radius**2)  # Rotor area (m^2)

    # Compute CT and CP
    CT = total_thrust / (0.5 * density * wind_speed**2 * rotor_area)
    CP = total_power / (0.5 * density * wind_speed**3 * rotor_area)

    return CT, CP


class Plot_results:
    """
    A class to handle plotting of power, thrust, and coefficients.
    """

    @staticmethod
    def Plot_Power_Thrust(wind_speed, total_thrust, total_power):
        """
        Plots the power and thrust curves as subplots in a single figure.

        Args:
            wind_speed (numpy.ndarray): Array of wind speeds.
            total_thrust (numpy.ndarray): Array of total thrust values.
            total_power (numpy.ndarray): Array of total power values.
        """

        # Create a figure with 2 subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot thrust on the first subplot
        ax1.set_xlabel('Wind Speed (m/s)')
        ax1.set_ylabel('Thrust (MN)', color='tab:blue')
        ax1.plot(wind_speed, total_thrust / 1e6, color='tab:blue',
                 label='Thrust')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)
        ax1.set_title('Thrust vs Wind Speed')

        # Plot power on the second subplot
        ax2.set_xlabel('Wind Speed (m/s)')
        ax2.set_ylabel('Power (MW)', color='tab:red')
        ax2.plot(wind_speed, total_power / 1e6, color='tab:red', label='Power')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.grid(True)
        ax2.set_title('Power vs Wind Speed')

        # Adjust layout for better spacing
        plt.tight_layout()

        return fig, (ax1, ax2)

    @staticmethod
    def Plot_CT_CP(wind_speed, c_thrust, c_power):
        """
        Plots the thrust coefficient (CT) and power coefficient (CP) as
        subplots in a single figure.

        Args:
            wind_speed (numpy.ndarray): Array of wind speeds.
            c_thrust (numpy.ndarray): Array of thrust coefficients.
            c_power (numpy.ndarray): Array of power coefficients.

        Returns:
            tuple: A tuple containing the figure and axes objects.
        """

        # Create a figure with 2 subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot CT on the first subplot
        ax1.set_xlabel('Wind Speed (m/s)')
        ax1.set_ylabel('CT', color='tab:blue')
        ax1.plot(wind_speed, c_thrust, color='tab:blue', label='CT')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)
        ax1.set_title('CT vs Wind Speed')

        # Plot CP on the second subplot
        ax2.set_xlabel('Wind Speed (m/s)')
        ax2.set_ylabel('CP', color='tab:red')
        ax2.plot(wind_speed, c_power, color='tab:red', label='CP')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.grid(True)
        ax2.set_title('CP vs Wind Speed')

        # Adjust layout for better spacing
        plt.tight_layout()

        return fig, (ax1, ax2)


def plot_CP_CT_TSR(tsr, cp, ct):
    """
    Plots CP vs TSR and CT vs TSR as subplots in a single figure.

    Args:
        tsr (list or numpy.ndarray): Array of Tip Speed Ratios (TSR).
        cp (list or numpy.ndarray): Array of Power Coefficients (CP).
        ct (list or numpy.ndarray): Array of Thrust Coefficients (CT).

    Returns:
        tuple: A tuple containing the figure and axes objects.
    """
    # Arrange TSR, CP, and CT values in ascending order of TSR
    sorted_indices = np.argsort(tsr)
    tsr = np.array(tsr)[sorted_indices]
    cp = np.array(cp)[sorted_indices]
    ct = np.array(ct)[sorted_indices]

    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot CP vs TSR on the first subplot
    ax1.plot(tsr, cp, color='tab:blue', label='CP vs TSR')
    ax1.set_xlabel('TSR')
    ax1.set_ylabel('CP', color='tab:blue')
    ax1.grid(True)
    ax1.set_title('CP vs TSR')

    # Plot CT vs TSR on the second subplot
    ax2.plot(tsr, ct, color='tab:red', label='CT vs TSR')
    ax2.set_xlabel('TSR')
    ax2.set_ylabel('CT', color='tab:red')
    ax2.grid(True)
    ax2.set_title('CT vs TSR')

    # Adjust layout for better spacing
    plt.tight_layout()

    return fig, (ax1, ax2)


def Plot_Power_Thrust_Compare(wind_speed, total_thrust, total_power, opt_data):
    """
    Plots the power and thrust curves in separate plots, comparing computed
    values with optimization data.

    Args:
        wind_speed (numpy.ndarray): Array of wind speeds.
        total_thrust (numpy.ndarray): Array of total thrust values.
        total_power (numpy.ndarray): Array of total power values.
        opt_data (dict): Dictionary containing optimization data with keys '
                         wind speed [m/s]','aero thrust [kn]', and
                         'aero power [kw]'.
    """

    # Extract optimization data
    opt_wind_speed = opt_data['wind speed [m/s]']
    opt_thrust = np.array(opt_data['aero thrust [kn]']) * 1e3  # Convert kN - N
    opt_power = np.array(opt_data['aero power [kw]']) * 1e3    # Convert kW - W

    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot thrust on the first subplot
    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('Thrust (MN)', color='tab:blue')
    ax1.plot(wind_speed, total_thrust / 1e6, color='tab:blue',
             label='Computed Thrust')
    ax1.plot(opt_wind_speed, opt_thrust / 1e6, color='tab:orange',
             linestyle='--', label='Observed Thrust')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Thrust vs Wind Speed')

    # Plot power on the second subplot
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Power (MW)', color='tab:red')
    ax2.plot(wind_speed, total_power / 1e6, color='tab:red',
             label='Computed Power')
    ax2.plot(opt_wind_speed, opt_power / 1e6, color='tab:green',
             linestyle='--', label='Observed Power')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Power vs Wind Speed')

    # Adjust layout for better spacing
    plt.tight_layout()

    return fig, (ax1, ax2)


class Corrected_ind_factors:
    """
    A class to compute corrected induction factors and compare power and
    thrust models.
    """

    @staticmethod
    def Compute_ind_factor_corrected(
        wind_speed, rot_speed, pitch_angle, blade_data, cl_table,
        cd_table, B=3, R=None
    ):
        """
        Computes the induction factor with Prandtl tip loss and Glauert
        correction.

        Args:
            wind_speed (float): The wind speed in m/s.
            rot_speed (float): The rotational speed in rad/s.
            pitch_angle (float): The pitch angle in degrees.
            blade_data (dict): A dictionary containing blade aerodynamic
            properties.
            cl_table (dict): A dictionary containing lift coefficient data.
            cd_table (dict): A dictionary containing drag coefficient data.
            B (int, optional): Number of blades. Defaults to 3.
            R (float, optional): Rotor radius. Defaults to None.

        Returns:
            tuple: A tuple containing two numpy arrays:
            - a: Corrected axial induction factors for each blade section.
            - a_prime: Corrected tangential induction factors for each blade
            section.
        """

        # Extract blade geometry and aerodynamic properties
        r_values = blade_data['blade_span_m']  # Radial positions
        c_values = blade_data['chord_length_m']  # Chord lengths
        beta_values = blade_data['twist_angle_deg']  # Twist angles

        # Determine rotor radius if not provided
        if R is None:
            R = np.max(r_values)  # Max radial position as radius

        # Initialize variables for induction factors
        n_sections = len(r_values)  # Number of blade sections
        a = np.zeros(n_sections)  # Axial induction factors
        a_prime = np.zeros(n_sections)  # Tangential factors
        a_new = np.ones(n_sections) * 0.01  # Initial guess for a
        a_prime_new = np.ones(n_sections) * 0.01  # Initial guess

        # Iterate over each blade section
        for i in range(n_sections):
            r = r_values[i]  # Radial position
            if r == 0:
                r = 1e-6  # Avoid division by zero
            c = c_values[i]  # Chord length
            beta = np.radians(beta_values[i])  # Twist angle in rad
            pitch = np.radians(pitch_angle)  # Pitch angle in rad

            # Initialize convergence criteria
            diff_a = np.inf  # Difference in axial induction factor
            diff_a_prime = np.inf  # Difference in tangential factor
            iteration_count = 0  # Iteration counter
            max_iterations = 500  # Max iterations allowed

            # Iterative process to compute induction factors
            while (
                diff_a > 1e-4 or diff_a_prime > 1e-4
                    ) and iteration_count < max_iterations:
                # Compute flow angle (phi)
                num = (1 - a[i]) * wind_speed
                denom = (1 + a_prime[i]) * rot_speed * r
                epsilon = 1e-6  # Small value to prevent division by zero
                phi = np.arctan(num / (denom + epsilon))

                # Compute angle of attack (aoa)
                aoa = phi - (beta + pitch)

                # Interpolate lift (cl) and drag (cd) coefficients
                cl_key = (
                    list(cl_table.keys())[i + 1]
                    if i + 1 < len(cl_table.keys())
                    else list(cl_table.keys())[-1]
                )
                cd_key = (
                    list(cd_table.keys())[i + 1]
                    if i + 1 < len(cd_table.keys())
                    else list(cd_table.keys())[-1]
                )
                cl = np.interp(np.degrees(aoa), cl_table['A0A'],
                               cl_table[cl_key])
                cd = np.interp(np.degrees(aoa), cd_table['A0A'],
                               cd_table[cd_key])

                # Compute normal (cn) and tangential (ct) coefficients
                cn = cl * np.cos(phi) + cd * np.sin(phi)
                ct = cl * np.sin(phi) - cd * np.cos(phi)

                # Compute solidity (sigma)
                sigma = B * c / (2 * np.pi * r)

                # ---- Prandtl Tip Loss Factor ----
                f = (B / 2) * (R - r) / (r * np.sin(phi))
                F = (
                    (2 / np.pi) * np.arccos(np.exp(-f))
                    if f > 1e-6
                    else 1.0
                )
                F = max(F, 1e-4)  # Prevent F from being zero

                # ---- Glauert Correction ----
                k = 4 * F * (np.sin(phi) ** 2) / (sigma * cn)
                a_temp = 1 / (k + 1)
                CT = sigma * cn * (a_temp ** 2) / (np.sin(phi) ** 2)

                # Apply Glauert correction for high thrust coefficients
                if CT > 0.96:
                    a_temp = (
                                18 * F
                                - 20
                                - 3
                                * np.sqrt(
                                    CT * (50 - 36 * F) + 12 * F * (3 * F - 4)
                                         )
                                 ) / (36 * F - 50)

                # Update induction factors
                a_new[i] = a_temp
                a_prime_new[i] = 1 / (
                    (4 * F * np.sin(phi) * np.cos(phi)) / (sigma * ct) - 1
                )

                # Convergence checks
                diff_a = np.abs(a_new[i] - a[i])
                diff_a_prime = np.abs(a_prime_new[i] - a_prime[i])
                a[i] = a_new[i]
                a_prime[i] = a_prime_new[i]
                iteration_count += 1

        return a, a_prime

    @staticmethod
    def plot_compare_Power_Thrust_models(total_thrust, total_power,
                                         total_thrust_corrected,
                                         total_power_corrected, wind_speed):
        """
        Plots the power and thrust curves in separate plots, comparing
        computed values with optimization data.

        Args:
            wind_speed (numpy.ndarray): Array of wind speeds.
            total_thrust (numpy.ndarray): Array of total thrust values.
            total_power (numpy.ndarray): Array of total power values.
            opt_data (dict): Dictionary containing optimization data with keys
                             'wind speed [m/s]','aero thrust [kn]', and
                            'aero power [kw]'.
        """

        # Filter data for wind speeds greater than 5
        mask = wind_speed > 5
        wind_speed = wind_speed[mask]
        total_thrust = total_thrust[mask]
        total_power = total_power[mask]
        total_thrust_corrected = total_thrust_corrected[mask]
        total_power_corrected = total_power_corrected[mask]

        # Create a figure with 2 subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot thrust on the first subplot
        ax1.set_xlabel('Wind Speed (m/s)')
        ax1.set_ylabel('Thrust (MN)', color='tab:blue')
        ax1.plot(wind_speed, total_thrust / 1e6, color='tab:blue',
                 label='Original Thrust')
        ax1.plot(wind_speed, total_thrust_corrected / 1e6,
                 color='tab:orange', linestyle='--', label='Corrected Thrust')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Thrust vs Wind Speed')

        # Plot power on the second subplot
        ax2.set_xlabel('Wind Speed (m/s)')
        ax2.set_ylabel('Power (MW)', color='tab:red')
        ax2.plot(wind_speed, total_power / 1e6, color='tab:red',
                 label='Original Power')
        ax2.plot(wind_speed, total_power_corrected / 1e6, color='tab:green',
                 linestyle='--', label='Corrected Power')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('Power vs Wind Speed')

        # Adjust layout for better spacing
        plt.tight_layout()

        return fig, (ax1, ax2)

def Plot_rotspeed_pitch(wind_speed, pitch, rot_speed):
    """
    Plots pitch angle and rotational speed as functions of wind speed.

    Args:
        wind_speed (list or numpy.ndarray): Array of wind speeds.
        pitch (list or numpy.ndarray): Array of pitch angles (in degrees).
        rot_speed (list or numpy.ndarray): Array of rotational speeds (in rad/s).

    Returns:
        tuple: A tuple containing the figure and axes objects.
    """
    # Convert rotational speed from rad/s to rpm for better readability
    rot_speed_rpm = np.array(rot_speed) * 60 / (2 * np.pi)

    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot pitch angle on the first subplot
    ax1.plot(wind_speed, pitch, color='tab:blue', label='Pitch Angle')
    ax1.set_xlabel('Wind Speed (m/s)')
    ax1.set_ylabel('Pitch Angle (deg)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    ax1.set_title('Pitch Angle vs Wind Speed')

    # Plot rotational speed on the second subplot
    ax2.plot(wind_speed, rot_speed_rpm, color='tab:red', label='Rotational Speed')
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Rotational Speed (rpm)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.grid(True)
    ax2.set_title('Rotational Speed vs Wind Speed')

    # Adjust layout for better spacing
    plt.tight_layout()

    return fig, (ax1, ax2)