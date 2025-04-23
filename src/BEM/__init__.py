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



def Read_Blade_data(file_path):
    """
    Reads the blade aerodynamic data from the specified file and organizes it into a structured format.

    Args:
        file_path (str): The path to the directory containing the blade data file.

    Returns:
        dict: A dictionary containing blade aerodynamic properties such as span, chord length, twist angle,
        airfoil IDs, and other related parameters.
    """
    # Define the blade data file name
    blade_file = os.path.join(file_path, 'IEA-15-240-RWT_AeroDyn15_blade.dat')

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
        grouped_cl = {file: [] for file in self.file_list}

        for file, file_data in self.data.items():
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
        grouped_cd = {file: [] for file in self.file_list}

        for file, file_data in self.data.items():
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
