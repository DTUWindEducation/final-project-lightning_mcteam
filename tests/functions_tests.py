import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as pl
import sys
import BEM
import pandas as pd
import pytest

import tempfile

# Paths

INPUT_PATH = pl.Path(__file__).resolve().parent.parent / "inputs"
Blade_characteristics_path = INPUT_PATH / "IEA-15-240-RWT"
Airfoil_Aerodynamic_path = INPUT_PATH / "IEA-15-240-RWT/Airfoils"
Airfoil_coord_path = INPUT_PATH / "IEA-15-240-RWT/Airfoils"
Operational_characteristics_path = INPUT_PATH / "IEA-15-240-RWT"

# File prefixes

Aifoil_Aerodynamic_file_prefix = "IEA-15-240-RWT_AeroDyn15_Polar_"
Aifoil_coord_file_prefix = "IEA-15-240-RWT_AF"



def test_Aerodynamic_file_names():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock files in the temporary directory
        filenames = [
            "IEA-15-240-RWT_AeroDyn15_Polar_01.dat",
            "IEA-15-240-RWT_AeroDyn15_Polar_02.dat",
            "IEA-15-240-RWT_AeroDyn15_Polar_03.dat",
            "unrelated_file.txt",
            "IEA-15-240-RWT_AeroDyn15_Polar_04.txt",
        ]
        for filename in filenames:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("Test content")

        # Call the function with the temporary directory and prefix
        result = BEM.Aerodynamic_file_names(temp_dir, "IEA-15-240-RWT_AeroDyn15_Polar_")

        # Expected result
        expected = [
            "IEA-15-240-RWT_AeroDyn15_Polar_01.dat",
            "IEA-15-240-RWT_AeroDyn15_Polar_02.dat",
            "IEA-15-240-RWT_AeroDyn15_Polar_03.dat",
        ]

        # Assert the result matches the expected output
        assert result == expected, f"Expected {expected}, but got {result}"


def test_Read_Blade_data():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define mock blade data content
        mock_blade_data = """\
        # Mock header line 1
        # Mock header line 2
        # Mock header line 3
        # Mock header line 4
        # Mock header line 5
        # Mock header line 6
        0.0 0.0 0.0 0.0 0.0 3.0 1 0.0 0.0 0.0
        5.0 0.1 0.1 1.0 2.0 3.5 2 0.1 0.1 0.1
        10.0 0.2 0.2 2.0 4.0 4.0 3 0.2 0.2 0.2
        """
        # Write mock blade data to a temporary file
        blade_file_path = os.path.join(temp_dir, "IEA-15-240-RWT_AeroDyn15_blade.dat")
        with open(blade_file_path, 'w') as blade_file:
            blade_file.write(mock_blade_data)

        # Call the function with the temporary file path
        result = BEM.Read_Blade_data(temp_dir)

        # Expected result
        expected = {
            'blade_span_m': [0.0, 5.0, 10.0],
            'curve_aero_center_m': [0.0, 0.1, 0.2],
            'sweep_aero_center_m': [0.0, 0.1, 0.2],
            'curve_angle_deg': [0.0, 1.0, 2.0],
            'twist_angle_deg': [0.0, 2.0, 4.0],
            'chord_length_m': [3.0, 3.5, 4.0],
            'airfoil_id': [1, 2, 3],
            'control_blend': [0.0, 0.1, 0.2],
            'center_bend_m': [0.0, 0.1, 0.2],
            'center_torsion_m': [0.0, 0.1, 0.2],
        }

        # Assert the result matches the expected output
        assert result == expected, f"Expected {expected}, but got {result}"


def test_Blade_order_Airfoils():
    # Mock blade data
    blade_data = {
        'airfoil_id': [3, 1, 2]
    }

    # Mock airfoil data
    airfoil_data = [
        "IEA-15-240-RWT_AeroDyn15_Polar_00.dat",
        "IEA-15-240-RWT_AeroDyn15_Polar_01.dat",
        "IEA-15-240-RWT_AeroDyn15_Polar_02.dat",
        "IEA-15-240-RWT_AeroDyn15_Polar_03.dat"
    ]

    # Call the function
    result = BEM.Blade_order_Airfoils(blade_data, airfoil_data)

    # Expected result
    expected = [
        "IEA-15-240-RWT_AeroDyn15_Polar_02.dat",
        "IEA-15-240-RWT_AeroDyn15_Polar_00.dat",
        "IEA-15-240-RWT_AeroDyn15_Polar_01.dat"
    ]

    # Assert the result matches the expected output
    assert result == expected, f"Expected {expected}, but got {result}"


def test_Aerodynamic_inputs_group_data_cl():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock aerodynamic data files
        file_1_content = "alpha cl cd\n0.0 0.1 0.01\n5.0 0.2 0.02\n10.0 0.3 0.03\n"
        file_2_content = "alpha cl cd\n0.0 0.15 0.015\n5.0 0.25 0.025\n10.0 0.35 0.035\n"
        file_3_content = "alpha cl cd\n0.0 0.12 0.012\n5.0 0.22 0.022\n10.0 0.32 0.032\n"

        filenames = ["file_1.dat", "file_2.dat", "file_3.dat"]
        file_contents = [file_1_content, file_2_content, file_3_content]

        for filename, content in zip(filenames, file_contents):
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write(content)

        # Initialize Aerodynamic_inputs with the mock files
        aerodynamic_inputs = BEM.Aerodynamic_inputs(filenames, temp_dir)

        # Call group_data_cl
        grouped_cl = aerodynamic_inputs.group_data_cl()

        # Expected result
        expected_cl = {
            'A0A': [0.0, 5.0, 10.0],
            'file_1.dat': [0.1, 0.2, 0.3],
            'file_2.dat': [0.15, 0.25, 0.35],
            'file_3.dat': [0.12, 0.22, 0.32]
        }

        # Assert the result matches the expected output
        assert grouped_cl == expected_cl, f"Expected {expected_cl}, but got {grouped_cl}"


def test_Aerodynamic_inputs_group_data_cd():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock aerodynamic data files
        file_1_content = "alpha cl cd\n0.0 0.1 0.01\n5.0 0.2 0.02\n10.0 0.3 0.03\n"
        file_2_content = "alpha cl cd\n0.0 0.15 0.015\n5.0 0.25 0.025\n10.0 0.35 0.035\n"
        file_3_content = "alpha cl cd\n0.0 0.12 0.012\n5.0 0.22 0.022\n10.0 0.32 0.032\n"

        filenames = ["file_1.dat", "file_2.dat", "file_3.dat"]
        file_contents = [file_1_content, file_2_content, file_3_content]

        for filename, content in zip(filenames, file_contents):
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write(content)

        # Initialize Aerodynamic_inputs with the mock files
        aerodynamic_inputs = BEM.Aerodynamic_inputs(filenames, temp_dir)

        # Call group_data_cd
        grouped_cd = aerodynamic_inputs.group_data_cd()

        # Expected result
        expected_cd = {
            'A0A': [0.0, 5.0, 10.0],
            'file_1.dat': [0.01, 0.02, 0.03],
            'file_2.dat': [0.015, 0.025, 0.035],
            'file_3.dat': [0.012, 0.022, 0.032]
        }

        # Assert the result matches the expected output
        assert grouped_cd == expected_cd, f"Expected {expected_cd}, but got {grouped_cd}"


def test_Airfoil_coord_names():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock files in the temporary directory
        filenames = [
            "IEA-15-240-RWT_AF01.txt",
            "IEA-15-240-RWT_AF02.txt",
            "IEA-15-240-RWT_AF03.txt",
            "unrelated_file.txt",
            "IEA-15-240-RWT_AF04.dat",
        ]
        for filename in filenames:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("Test content")

        # Call the function with the temporary directory and prefix
        result = BEM.Airfoil_coord_names(temp_dir, "IEA-15-240-RWT_AF")

        # Expected result
        expected = [
            "IEA-15-240-RWT_AF01.txt",
            "IEA-15-240-RWT_AF02.txt",
            "IEA-15-240-RWT_AF03.txt",
        ]

        # Assert the result matches the expected output
        assert result == expected, f"Expected {expected}, but got {result}"


def test_plot_airfoil():
    # Create a temporary directory for mock airfoil files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock airfoil file content
        airfoil_content = """\
        0.0 0.0
        0.5 0.1
        1.0 0.0
        0.5 -0.1
        0.0 0.0
        """
        # Create mock airfoil files
        airfoil_file_names = []
        for i in range(3):
            file_name = f"airfoil_{i}.dat"
            airfoil_file_names.append(file_name)
            with open(os.path.join(temp_dir, file_name), 'w') as f:
                f.write(airfoil_content)

        # Mock blade data
        blade_data = {
            'blade_span_m': [0.0, 5.0, 10.0],
            'twist_angle_deg': [0.0, 10.0, -10.0]
        }

        # Call the function
        fig, ax = BEM.plot_airfoil(airfoil_file_names, temp_dir, blade_data)

        # Assertions
        assert isinstance(fig, plt.Figure), "The returned figure is not a matplotlib Figure."
        assert ax.name == '3d', "The returned axes is not a 3D Axes."
        assert len(ax.lines) == len(airfoil_file_names), "The number of plotted lines does not match the number of airfoil files."

        # Clean up the plot
        plt.close(fig)


def test_Blade_opt_data():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define mock blade optimization data content
        mock_opt_data = """\
wind speed [m/s] pitch [deg] rot. speed [rpm] aero power [kw] aero thrust [kn]
5.0 2.0 10.0 500.0 50.0
10.0 3.0 15.0 1000.0 100.0
15.0 4.0 20.0 1500.0 150.0
"""
        # Write mock data to a temporary file
        opt_file_path = os.path.join(temp_dir, "IEA_15MW_RWT_Onshore.opt")
        with open(opt_file_path, 'w') as opt_file:
            opt_file.write(mock_opt_data)

        # Call the function with the temporary file path
        result = BEM.Blade_opt_data(temp_dir, "IEA_15MW_RWT_Onshore.opt")

        # Expected result
        expected = {
            'wind speed [m/s]': [5.0, 10.0, 15.0],
            'pitch [deg]': [2.0, 3.0, 4.0],
            'rot. speed [rpm]': [10.0, 15.0, 20.0],
            'aero power [kw]': [500.0, 1000.0, 1500.0],
            'aero thrust [kn]': [50.0, 100.0, 150.0]
        }

        # Assert the result matches the expected output
        assert result == expected, f"Expected {expected}, but got {result}"


def test_Compute_TSR_pitch():
    # Mock input data
    wind_speed = 10.0
    dict_opt_data = {
        'wind speed [m/s]': [5.0, 10.0, 15.0],
        'pitch [deg]': [2.0, 3.0, 4.0],
        'rot. speed [rpm]': [10.0, 15.0, 20.0]
    }
    rotor_radius = 120

    # Call the function
    tsr, pitch, rot_speed = BEM.Compute_TSR_pitch(wind_speed, dict_opt_data, rotor_radius)

    # Expected results
    expected_tsr = (15.0 * 2 * 3.141592653589793 / 60) * rotor_radius / wind_speed
    expected_pitch = 3.0
    expected_rot_speed = 15.0 * 2 * 3.141592653589793 / 60

    # Assertions
    assert pytest.approx(tsr, rel=1e-3) == expected_tsr, f"Expected TSR {expected_tsr}, but got {tsr}"
    assert pytest.approx(pitch, rel=1e-3) == expected_pitch, f"Expected pitch {expected_pitch}, but got {pitch}"
    assert pytest.approx(rot_speed, rel=1e-3) == expected_rot_speed, f"Expected rotational speed {expected_rot_speed}, but got {rot_speed}"


def test_Compute_ind_factor():
    # Mock input data
    wind_speed = 10.0
    rot_speed = 1.0
    pitch_angle = 2.0
    blade_data = {
        'blade_span_m': [5.0, 10.0, 15.0],
        'chord_length_m': [1.0, 1.2, 1.5],
        'twist_angle_deg': [5.0, 10.0, 15.0]
    }
    cl_table = {
        'A0A': [0.0, 5.0, 10.0],
        'cl_1': [0.1, 0.2, 0.3],
        'cl_2': [0.15, 0.25, 0.35],
        'cl_3': [0.2, 0.3, 0.4]
    }
    cd_table = {
        'A0A': [0.0, 5.0, 10.0],
        'cd_1': [0.01, 0.02, 0.03],
        'cd_2': [0.015, 0.025, 0.035],
        'cd_3': [0.02, 0.03, 0.04]
    }
    B = 3

    # Call the function
    a, a_prime = BEM.Compute_ind_factor(wind_speed, rot_speed, pitch_angle, blade_data, cl_table, cd_table, B)

    # Expected results (mocked for simplicity)
    expected_a = np.array([0.01, 0.01, 0.01])  # Initial guess
    expected_a_prime = np.array([0.01, 0.01, 0.01])  # Initial guess

    # Assertions
    assert len(a) == len(blade_data['blade_span_m']), "Length of 'a' does not match blade span data."
    assert len(a_prime) == len(blade_data['blade_span_m']), "Length of 'a_prime' does not match blade span data."
    assert np.allclose(a, expected_a, atol=1e-2), f"Expected 'a' close to {expected_a}, but got {a}"
    assert np.allclose(a_prime, expected_a_prime, atol=1e-2), f"Expected 'a_prime' close to {expected_a_prime}, but got {a_prime}"


def test_Compute_local_thrust_moemnt():
    # Mock input data
    a = np.array([0.2, 0.3, 0.4])
    a_prime = np.array([0.1, 0.15, 0.2])
    wind_speed = 10.0
    rot_speed = 1.0
    blade_data = {
        'blade_span_m': [5.0, 10.0, 15.0]
    }
    density = 1.225

    # Call the function
    thrust, moment = BEM.Compute_local_thrust_moemnt(a, a_prime, wind_speed, rot_speed, blade_data, density)

    # Expected results
    expected_thrust = np.array([
        4 * np.pi * 5.0 * density * (wind_speed ** 2) * a[0] * (1 - a[0]),
        4 * np.pi * 10.0 * density * (wind_speed ** 2) * a[1] * (1 - a[1]),
        4 * np.pi * 15.0 * density * (wind_speed ** 2) * a[2] * (1 - a[2])
    ])
    expected_moment = np.array([
        4 * np.pi * (5.0 ** 3) * density * wind_speed * rot_speed * a_prime[0] * (1 - a[0]),
        4 * np.pi * (10.0 ** 3) * density * wind_speed * rot_speed * a_prime[1] * (1 - a[1]),
        4 * np.pi * (15.0 ** 3) * density * wind_speed * rot_speed * a_prime[2] * (1 - a[2])
    ])

    # Assertions
    assert np.allclose(thrust, expected_thrust, atol=1e-2), f"Expected thrust {expected_thrust}, but got {thrust}"
    assert np.allclose(moment, expected_moment, atol=1e-2), f"Expected moment {expected_moment}, but got {moment}"


def test_Compute_Power_Thrust():
    # Mock input data
    thrust = np.array([100.0, 200.0, 300.0])  # Local thrust values
    moment = np.array([50.0, 100.0, 150.0])   # Local moment values
    rot_speed = 2.0                           # Rotational speed in rad/s
    rated_power = 500.0                       # Rated power in watts
    blade_data = {
        'blade_span_m': [0.0, 5.0, 10.0]      # Radial segments
    }

    # Call the function
    total_thrust, total_power = BEM.Compute_Power_Thrust(thrust, moment, rot_speed, rated_power, blade_data)

    # Expected results
    expected_total_thrust = np.trapz(thrust, blade_data['blade_span_m'])
    expected_total_moment = np.trapz(moment, blade_data['blade_span_m'])
    expected_total_power = min(expected_total_moment * rot_speed, rated_power)

    # Assertions
    assert pytest.approx(total_thrust, rel=1e-3) == expected_total_thrust, \
        f"Expected total thrust {expected_total_thrust}, but got {total_thrust}"
    assert pytest.approx(total_power, rel=1e-3) == expected_total_power, \
        f"Expected total power {expected_total_power}, but got {total_power}"
    

def test_Compute_CT_CP():
    # Test case 1: Standard inputs
    total_thrust = 1000.0  # N
    total_power = 50000.0  # W
    wind_speed = 10.0  # m/s
    rot_radius = 50.0  # m
    density = 1.225  # kg/m^3

    CT, CP = BEM.Compute_CT_CP(total_thrust, total_power, wind_speed, rot_radius, density)

    rotor_area = 3.141592653589793 * (rot_radius ** 2)
    expected_CT = total_thrust / (0.5 * density * wind_speed ** 2 * rotor_area)
    expected_CP = total_power / (0.5 * density * wind_speed ** 3 * rotor_area)

    assert pytest.approx(CT, rel=1e-3) == expected_CT, f"Expected CT {expected_CT}, but got {CT}"
    assert pytest.approx(CP, rel=1e-3) == expected_CP, f"Expected CP {expected_CP}, but got {CP}"

    # Test case 2: Zero wind speed
    wind_speed = 0.0
    with pytest.raises(ZeroDivisionError):
        BEM.Compute_CT_CP(total_thrust, total_power, wind_speed, rot_radius, density)

    # Test case 3: Zero rotor radius
    rot_radius = 0.0
    with pytest.raises(ZeroDivisionError):
        BEM.Compute_CT_CP(total_thrust, total_power, wind_speed, rot_radius, density)


def test_Plot_Power_Thrust():
    # Mock input data
    wind_speed = np.array([5, 10, 15, 20])
    total_thrust = np.array([1000000, 2000000, 3000000, 4000000])  # in N
    total_power = np.array([500000, 1000000, 1500000, 2000000])    # in W

    # Call the function
    (fig1, ax1), (fig2, ax2) = BEM.Plot_Power_Thrust(wind_speed, total_thrust, total_power)

    # Assertions for thrust plot
    assert isinstance(fig1, plt.Figure), "Thrust plot figure is not a matplotlib Figure."
    assert ax1.get_xlabel() == 'Wind Speed (m/s)', "Thrust plot x-axis label is incorrect."
    assert ax1.get_ylabel() == 'Thrust (MN)', "Thrust plot y-axis label is incorrect."
    assert len(ax1.lines) == 1, "Thrust plot should have one line."
    assert np.array_equal(ax1.lines[0].get_xdata(), wind_speed), "Thrust plot x-data is incorrect."
    assert np.array_equal(ax1.lines[0].get_ydata(), total_thrust / 1e6), "Thrust plot y-data is incorrect."

    # Assertions for power plot
    assert isinstance(fig2, plt.Figure), "Power plot figure is not a matplotlib Figure."
    assert ax2.get_xlabel() == 'Wind Speed (m/s)', "Power plot x-axis label is incorrect."
    assert ax2.get_ylabel() == 'Power (MW)', "Power plot y-axis label is incorrect."
    assert len(ax2.lines) == 1, "Power plot should have one line."
    assert np.array_equal(ax2.lines[0].get_xdata(), wind_speed), "Power plot x-data is incorrect."
    assert np.array_equal(ax2.lines[0].get_ydata(), total_power / 1e6), "Power plot y-data is incorrect."

    # Clean up the plots
    plt.close(fig1)
    plt.close(fig2)


def test_Plot_CT_CP():
    # Mock input data
    wind_speed = np.array([5, 10, 15, 20])
    c_thrust = np.array([0.1, 0.2, 0.3, 0.4])
    c_power = np.array([0.05, 0.15, 0.25, 0.35])

    # Call the function
    (fig1, ax1), (fig2, ax2) = BEM.Plot_CT_CP(wind_speed, c_thrust, c_power)

    # Assertions for CT plot
    assert isinstance(fig1, plt.Figure), "CT plot figure is not a matplotlib Figure."
    assert ax1.get_xlabel() == 'Wind Speed (m/s)', "CT plot x-axis label is incorrect."
    assert ax1.get_ylabel() == 'CT', "CT plot y-axis label is incorrect."
    assert len(ax1.lines) == 1, "CT plot should have one line."
    assert np.array_equal(ax1.lines[0].get_xdata(), wind_speed), "CT plot x-data is incorrect."
    assert np.array_equal(ax1.lines[0].get_ydata(), c_thrust), "CT plot y-data is incorrect."

    # Assertions for CP plot
    assert isinstance(fig2, plt.Figure), "CP plot figure is not a matplotlib Figure."
    assert ax2.get_xlabel() == 'Wind Speed (m/s)', "CP plot x-axis label is incorrect."
    assert ax2.get_ylabel() == 'CP', "CP plot y-axis label is incorrect."
    assert len(ax2.lines) == 1, "CP plot should have one line."
    assert np.array_equal(ax2.lines[0].get_xdata(), wind_speed), "CP plot x-data is incorrect."
    assert np.array_equal(ax2.lines[0].get_ydata(), c_power), "CP plot y-data is incorrect."

    # Clean up the plots
    plt.close(fig1)
    plt.close(fig2)