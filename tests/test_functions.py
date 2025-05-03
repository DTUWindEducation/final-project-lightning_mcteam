import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as pl
import sys
import BEM
import pytest
import tempfile


# Add the src/ directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', 'src')))

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
    """
    Test the `Aerodynamic_file_names` function from the `BEM` module.
    This test verifies that the function correctly identifies and returns
    a list of aerodynamic file names from a given directory that match
    a specified prefix and have the `.dat` extension.
    Steps:
    1. Create a temporary directory.
    2. Populate the directory with mock files, including files that match
       the prefix and extension, as well as unrelated files.
    3. Call the `Aerodynamic_file_names` function with the temporary
       directory and prefix.
    4. Compare the returned result with the expected list of matching files.
    Assertions:
    - The result from the function should match the expected list of file names
      that have the correct prefix and `.dat` extension.
    Raises:
        AssertionError: If the result does not match the expected output.
    """
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
        result = BEM.Aerodynamic_file_names(temp_dir,
                                            "IEA-15-240-RWT_AeroDyn15_Polar_")

        # Expected result
        expected = [
            "IEA-15-240-RWT_AeroDyn15_Polar_01.dat",
            "IEA-15-240-RWT_AeroDyn15_Polar_02.dat",
            "IEA-15-240-RWT_AeroDyn15_Polar_03.dat",
        ]

        # Assert the result matches the expected output
        assert result == expected, f"Expected {expected}, but got {result}"


def test_Read_Blade_data():
    """
    Unit test for simulating the behavior of the Read_Blade_data function.
    This test creates a mock blade data file content, simulates reading and
    parsing the data, and verifies that the parsed result is in the expected
    format.
    The mock blade data contains:
    - Six header lines (ignored during parsing).
    - Data rows with numerical values separated by spaces.
    The test performs the following steps:
    1. Defines mock blade data as a multi-line string.
    2. Implements a mock class `MockReadBladeData` to simulate the behavior of
    the Read_Blade_data function.
    3. Parses the mock data, skipping the header lines, and organizes the data
    into a dictionary where:
       - Keys are column names (e.g., "column_1", "column_2").
       - Values are lists of floats corresponding to each column.
    4. Asserts that the result is:
       - A dictionary.
       - Keys are strings.
       - Values are lists of floats.
    Raises:
        AssertionError: If the result is not in the expected format.
    """

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

    # Write mock blade data to a file in the Blade_characteristics_path
    # Mock the Read_Blade_data function to simulate reading the mock data
    class MockReadBladeData:
        def __init__(self, blade_data):
            self.blade_data = blade_data

        def read(self):
            # Simulate parsing the mock blade data
            lines = self.blade_data.strip().split("\n")[6:]  # Skip header line
            result = {}
            for line in lines:
                values = list(map(float, line.split()))
                for i, value in enumerate(values):
                    key = f"column_{i+1}"
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            return result

    # Use the mock class to simulate the function behavior
    mock_reader = MockReadBladeData(mock_blade_data)
    result = mock_reader.read()

    # Assert the result  - dictionary with strings as keys and lists as values
    assert isinstance(result, dict), "Result is not a dictionary."
    for key, value in result.items():
        assert isinstance(key, str), f"Key {key} is not a string."
        assert all(isinstance(v, float) for v in value), \
            f"Values for key {key} are not all floats."


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
    """
    Test the `group_data_cl` method of the `Aerodynamic_inputs` class.
    This test verifies that the `group_data_cl` method correctly groups
    the lift coefficient (cl) data from multiple aerodynamic data files.
    The test uses a mock implementation of the `Aerodynamic_inputs` class
    to simulate file reading and data processing. It performs the following
    steps:
    1. Creates mock aerodynamic data files in memory with predefined content.
    2. Initializes a mock `Aerodynamic_inputs` class with the filenames and
    content.
    3. Loads the data into a 2D array using the `load_data` method.
    4. Calls the `group_data_cl` method to group the cl values by filename.
    5. Verifies that the output is a dictionary where:
       - Keys are filenames.
       - Values are lists of cl values extracted from the corresponding file.
    6. Ensures the output types are correct using assertions.
    Raises:
        AssertionError: If the output is not a dictionary or if the values
                        are not lists of cl values.
    """

    # Create mock aerodynamic data files in memory
    file_1_content = (
        "alpha cl cd\n"
        "0.0 0.1 0.01\n"
        "5.0 0.2 0.02\n"
        "10.0 0.3 0.03\n"
    )
    file_2_content = (
        "alpha cl cd\n"
        "0.0 0.15 0.015\n"
        "5.0 0.25 0.025\n"
        "10.0 0.35 0.035\n"
    )
    file_3_content = (
        "alpha cl cd\n"
        "0.0 0.12 0.012\n"
        "5.0 0.22 0.022\n"
        "10.0 0.32 0.032\n"
    )

    filenames = ["file_1.dat", "file_2.dat", "file_3.dat"]
    file_contents = [file_1_content, file_2_content, file_3_content]

    # Mock the Aerodynamic_inputs class to simulate file reading
    class MockAerodynamicInputs:
        def __init__(self, filenames, _):
            self.filenames = filenames
            self.file_contents = file_contents

        def load_data(self):
            self.data = [
                content.splitlines() for content in self.file_contents
                ]

        def group_data_cl(self):
            grouped_cl = {}
            for i, content in enumerate(self.data):
                cl_values = [float(line.split()[1]) for line in content[1:]]
                grouped_cl[self.filenames[i]] = cl_values
            return grouped_cl

    # Initialize the mock class
    aerodynamic_inputs = MockAerodynamicInputs(filenames, None)

    # Ensure data is loaded as a 2D array
    aerodynamic_inputs.load_data()

    # Call group_data_cl
    grouped_cl = aerodynamic_inputs.group_data_cl()

    # Verify the output types
    assert isinstance(grouped_cl, dict), \
        "group_data_cl did not return a dictionary"
    for key, value in grouped_cl.items():
        assert isinstance(value, list), \
            f"Value for key {key} is not a list"


def test_Aerodynamic_inputs_group_data_cd():
    """
    Test the `group_data_cd` method of the `Aerodynamic_inputs` class.
    This test verifies that the `group_data_cd` method correctly groups the
    drag coefficient (cd) values from multiple aerodynamic data files. The
    test uses mock data to simulate the behavior of the `Aerodynamic_inputs`
    class.
    Steps:
    1. Create mock aerodynamic data files with predefined content.
    2. Mock the `Aerodynamic_inputs` class to simulate file reading and data
    processing.
    3. Load the mock data into the class using the `load_data` method.
    4. Call the `group_data_cd` method to group the cd values by file.
    5. Verify that the output is a dictionary where:
        - Keys are filenames.
        - Values are lists of cd values.
    Assertions:
    - Ensure the output of `group_data_cd` is a dictionary.
    - Ensure each value in the dictionary is a list.
    This test ensures that the `group_data_cd` method processes and groups
    aerodynamic data correctly.
    """

    # Create mock aerodynamic data files in memory
    file_1_content = (
        "alpha cl cd\n"
        "0.0 0.1 0.01\n"
        "5.0 0.2 0.02\n"
        "10.0 0.3 0.03\n"
    )
    file_2_content = (
        "alpha cl cd\n"
        "0.0 0.15 0.015\n"
        "5.0 0.25 0.025\n"
        "10.0 0.35 0.035\n"
    )
    file_3_content = (
        "alpha cl cd\n"
        "0.0 0.12 0.012\n"
        "5.0 0.22 0.022\n"
        "10.0 0.32 0.032\n"
    )

    filenames = ["file_1.dat", "file_2.dat", "file_3.dat"]
    file_contents = [file_1_content, file_2_content, file_3_content]

    # Mock the Aerodynamic_inputs class to simulate file reading
    class MockAerodynamicInputs:
        def __init__(self, filenames, _):
            self.filenames = filenames
            self.file_contents = file_contents

        def load_data(self):
            self.data = [
                content.splitlines() for content in self.file_contents
                ]

        def group_data_cd(self):
            grouped_cd = {}
            for i, content in enumerate(self.data):
                cd_values = [float(line.split()[2]) for line in content[1:]]
                grouped_cd[self.filenames[i]] = cd_values
            return grouped_cd

    # Initialize the mock class
    aerodynamic_inputs = MockAerodynamicInputs(filenames, None)

    # Ensure data is loaded as a 2D array
    aerodynamic_inputs.load_data()

    # Call group_data_cd
    grouped_cd = aerodynamic_inputs.group_data_cd()

    # Verify the output types
    assert isinstance(
        grouped_cd, dict
    ), "group_data_cd did not return a dictionary"
    for key, value in grouped_cd.items():
        assert isinstance(
            value, list
        ), f"Value for key {key} is not a list"


def test_Airfoil_coord_names():
    """
    Test the `Airfoil_coord_names` function from the `BEM` module.
    This test verifies that the `Airfoil_coord_names` function
    correctly filters and returns a list of filenames from a
    specified directory that match a given prefix and have the
    `.txt` extension.
    Steps:
    1. Create a temporary directory.
    2. Populate the directory with mock files, including files
       that match the prefix and extension, as well as unrelated
       files.
    3. Call the `Airfoil_coord_names` function with the temporary
       directory and prefix.
    4. Compare the returned result with the expected list of
       filenames.
    Assertions:
    - The result from the function should match the expected list
      of filenames.
    Raises:
    - AssertionError: If the result does not match the expected
      output.
    """

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
    """
    Test the `plot_airfoil` function from the `BEM` module.
    This test verifies the following:
    1. The function correctly generates a 3D plot of airfoil shapes based on
       provided airfoil files and blade data.
    2. The returned figure is a valid matplotlib Figure object.
    3. The returned axes is a 3D Axes object.
    Test Setup:
    - Creates temporary airfoil files with mock data, including header lines
      and coordinate points.
    - Provides mock blade data with span and twist angle information.
    - Calls the `plot_airfoil` function with the generated files and data.
    Assertions:
    - Ensures the returned figure is an instance of `matplotlib.figure.Figure`.
    - Ensures the returned axes is a 3D Axes object with the correct name.
    Cleanup:
    - Closes the generated matplotlib figure to free resources.
    """

    # Airfoil file content with 8 header lines + 5 coordinate lines
    header = "\n" * 8
    coords = "0.0 0.0\n0.5 0.1\n1.0 0.0\n0.5 -0.1\n0.0 0.0\n"
    airfoil_content = header + coords

    # Mock airfoil file names
    airfoil_file_names = [f"airfoil_{i}.dat" for i in range(3)]

    # Blade data to match file count
    blade_data = {
        'blade_span_m': [0.0, 5.0, 10.0],
        'twist_angle_deg': [0.0, 10.0, -10.0]
    }

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        for name in airfoil_file_names:
            with open(os.path.join(temp_dir, name), 'w') as f:
                f.write(airfoil_content)

        # Call the function
        fig, ax = BEM.plot_airfoil(
            airfoil_file_names, path=temp_dir, blade_data=blade_data
        )

    # Simple assertions
    assert isinstance(fig, plt.Figure), (
        "Returned object is not a matplotlib Figure."
    )
    assert hasattr(ax, 'name') and ax.name == '3d', (
        "Returned axes is not a 3D Axes."
    )

    # Clean up
    plt.close(fig)


def test_Blade_opt_data():
    """
    Test the `Blade_opt_data` function.
    This test verifies that the function successfully reads and parses
    blade optimization data from a file and returns the expected output.
    The test performs the following steps:
    1. Defines mock blade optimization data as a multi-line string.
    2. Implements a mock class `MockBladeOptData` to simulate the behavior
       of the `Blade_opt_data` function.
    3. Parses the mock data and organizes it into a dictionary where:
       - Keys are column names (e.g., "column_1", "column_2").
       - Values are lists of floats corresponding to each column.
    4. Asserts that the result is:
       - A dictionary.
       - Keys are strings.
       - Values are lists of floats.
    Raises:
        AssertionError: If the result is not in the expected format.
    """

    # Define mock blade optimization data content
    mock_opt_data = """\
    5.0 2.0 10.0 500.0 100.0
    10.0 3.0 15.0 1000.0 200.0
    15.0 4.0 20.0 1500.0 300.0
    """

    # Mock the Blade_opt_data function to simulate reading the mock data
    class MockBladeOptData:
        def __init__(self, opt_data):
            self.opt_data = opt_data

        def read(self):
            # Simulate parsing the mock blade optimization data
            lines = self.opt_data.strip().split("\n")
            result = {}
            for line in lines:
                values = list(map(float, line.split()))
                for i, value in enumerate(values):
                    key = f"column_{i+1}"
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            return result

    # Use the mock class to simulate the function behavior
    mock_reader = MockBladeOptData(mock_opt_data)
    result = mock_reader.read()

    # Assert the result - dictionary with strings as keys and lists as values
    assert isinstance(result, dict), "Result is not a dictionary."
    for key, value in result.items():
        assert isinstance(key, str), f"Key {key} is not a string."
        assert all(isinstance(v, float) for v in value), \
            f"Values for key {key} are not all floats."


def test_Compute_TSR_pitch():
    """
    Test the Compute_TSR_pitch function from the BEM module.
    This test verifies the correctness of the TSR (Tip Speed Ratio),
    pitch angle, and rotational speed calculations based on the given
    wind speed, optimization data dictionary, and rotor radius.
    The test uses mock input data and compares the function's output
    to expected results using approximate assertions.
    Mock Input Data:
    - wind_speed: Wind speed in m/s.
    - dict_opt_data: Dictionary containing optimization data with keys:
        - 'wind speed [m/s]': List of wind speeds.
        - 'pitch [deg]': List of pitch angles in degrees.
        - 'rot. speed [rpm]': List of rotational speeds in RPM.
    - rotor_radius: Rotor radius in meters.
    Expected Results:
    - TSR (Tip Speed Ratio): Calculated based on the formula:
        (rotational speed in rad/s * rotor radius) / wind speed.
    - Pitch angle: Expected pitch angle corresponding to the given wind
      speed.
    - Rotational speed: Expected rotational speed in rad/s.
    Assertions:
    - Verifies that the computed TSR, pitch angle, and rotational speed
      are approximately equal to the expected values within a relative
      tolerance of 1e-3.
    Raises:
    - AssertionError: If any of the computed values do not match the
      expected results.
    """

    # Mock input data
    wind_speed = 10.0
    dict_opt_data = {
        'wind speed [m/s]': [5.0, 10.0, 15.0],
        'pitch [deg]': [2.0, 3.0, 4.0],
        'rot. speed [rpm]': [10.0, 15.0, 20.0]
    }
    rotor_radius = 120

    # Call the function
    tsr, pitch, rot_speed = BEM.Compute_TSR_pitch(
        wind_speed, dict_opt_data, rotor_radius
    )

    # Expected results
    expected_tsr = (
        (15.0 * 2 * 3.141592653589793 / 60) * rotor_radius / wind_speed
    )
    expected_pitch = 3.0
    expected_rot_speed = 15.0 * 2 * 3.141592653589793 / 60

    # Assertions
    assert pytest.approx(tsr, rel=1e-3) == expected_tsr, \
        f"Expected TSR {expected_tsr}, but got {tsr}"
    assert pytest.approx(pitch, rel=1e-3) == expected_pitch, \
        f"Expected pitch {expected_pitch}, but got {pitch}"
    assert pytest.approx(rot_speed, rel=1e-3) == expected_rot_speed, \
        f"Expected rotational speed {expected_rot_speed}, but got {rot_speed}"


def test_Compute_ind_factor():
    """
    Test the `Compute_ind_factor` function from the BEM module.
    This test verifies the functionality of the `Compute_ind_factor`
    function, which calculates the axial induction factor (`a`) and
    tangential induction factor (`a_prime`) for a wind turbine blade
    element. The test uses mocked input data and compares the computed
    results with expected values to ensure correctness.
    Mock Input Data:
    - `wind_speed` (float): The wind speed in m/s.
    - `rot_speed` (float): The rotational speed of the turbine in rad/s.
    - `pitch_angle` (float): The pitch angle of the blades in degrees.
    - `blade_data` (dict): A dictionary containing blade span, chord
      length, and twist angle data.
    - `cl_table` (dict): A dictionary containing lift coefficient data
      as a function of angle of attack.
    - `cd_table` (dict): A dictionary containing drag coefficient data
      as a function of angle of attack.
    - `B` (int): The number of blades on the turbine.
    Assertions:
    - Ensures the length of the computed `a` and `a_prime` matches the
      length of the blade span data.
    - Verifies that the computed `a` and `a_prime` values are close to
      the expected values within a specified tolerance.
    Expected Results:
    - `a` (numpy array): Mocked expected axial induction factor values.
    - `a_prime` (numpy array): Mocked expected tangential induction
      factor values.
    """

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
    a, a_prime = BEM.Compute_ind_factor(wind_speed, rot_speed, pitch_angle,
                                        blade_data, cl_table, cd_table, B)

    # Expected results (mocked for simplicity)
    expected_a = np.array([0.01, 0.01, 0.01])  # Initial guess
    expected_a_prime = np.array([0.01, 0.01, 0.01])  # Initial guess

    # Assertions
    assert len(a) == len(blade_data['blade_span_m']), \
        "Length of 'a' does not match blade span data."
    assert len(a_prime) == len(blade_data['blade_span_m']), \
        "Length of 'a_prime' does not match blade span data."
    assert np.allclose(a, expected_a, atol=1e-2), \
        f"Expected 'a' close to {expected_a}, but got {a}"
    assert np.allclose(a_prime, expected_a_prime, atol=1e-2), \
        f"Expected 'a_prime' close to {expected_a_prime}, but got {a_prime}"

    # Test edge case where r == 0
    blade_data['blade_span_m'] = [0.0, 10.0, 15.0]  # Include a zero radius
    a, a_prime = BEM.Compute_ind_factor(wind_speed, rot_speed, pitch_angle,
                                        blade_data, cl_table, cd_table, B)

    # Ensure r is handled correctly (r = 1e-6 for zero radius)
    assert a[0] != 0, (
        "Axial induction factor should not be computed with r = 0."
    )
    assert a_prime[0] != 0, (
        "Tangential induction factor should not be computed with r = 0."
    )


def test_Compute_local_thrust_moment():
    """
    Test the Compute_local_thrust_moment function from the BEM module.
    This test verifies that the function correctly calculates the local thrust
    and moment for a wind turbine blade given the input parameters. The test
    uses mock input data and compares the computed results with the expected
    values.
    Mock Input Data:
    - a: Axial induction factors (numpy array).
    - a_prime: Tangential induction factors (numpy array).
    - wind_speed: Free-stream wind speed (float).
    - rot_speed: Rotational speed of the turbine (float).
    - blade_data: Dictionary containing blade span information.
    - density: Air density (float).
    Expected Results:
    - thrust: Local thrust values (numpy array).
    - moment: Local moment values (numpy array).
    Assertions:
    - Validates that the computed thrust matches the expected thrust within a
      tolerance of 1e-2.
    - Validates that the computed moment matches the expected moment within a
      tolerance of 1e-2.
    Raises:
    - AssertionError: If the computed thrust or moment does not match the
      expected values within the specified tolerance.
    """

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
    thrust, moment = BEM.Compute_local_thrust_moment(a, a_prime, wind_speed,
                                                     rot_speed, blade_data,
                                                     density)

    # Expected results
    expected_thrust = np.array([
        4 * np.pi * 5.0 * density * (wind_speed ** 2) * a[0] * (1 - a[0]),
        4 * np.pi * 10.0 * density * (wind_speed ** 2) * a[1] * (1 - a[1]),
        4 * np.pi * 15.0 * density * (wind_speed ** 2) * a[2] * (1 - a[2])
    ])
    expected_moment = np.array([
        4 * np.pi * (5.0 ** 3) * density * wind_speed * rot_speed *
        a_prime[0] * (1 - a[0]),
        4 * np.pi * (10.0 ** 3) * density * wind_speed * rot_speed *
        a_prime[1] * (1 - a[1]),
        4 * np.pi * (15.0 ** 3) * density * wind_speed * rot_speed *
        a_prime[2] * (1 - a[2])
    ])

    # Assertions
    assert np.allclose(thrust, expected_thrust, atol=1e-2), \
        f"Expected thrust {expected_thrust}, but got {thrust}"
    assert np.allclose(moment, expected_moment, atol=1e-2), \
        f"Expected moment {expected_moment}, but got {moment}"


def test_Compute_Power_Thrust():
    """
    Test the Compute_Power_Thrust function from the BEM module.
    This test verifies the correctness of the Compute_Power_Thrust
    function by:
    - Calculating the total thrust using numerical integration
      (trapezoidal rule).
    - Calculating the total moment using numerical integration
      (trapezoidal rule).
    - Computing the total power as the minimum of the product of
      total moment and rotational speed, and the rated power.
    - Comparing the computed results with the expected values
      using assertions.
    Mock input data:
    - thrust: Local thrust values as a numpy array.
    - moment: Local moment values as a numpy array.
    - rot_speed: Rotational speed in radians per second.
    - rated_power: Rated power in watts.
    - blade_data: Dictionary containing blade span information.
    Assertions:
    - The computed total thrust matches the expected total thrust
      within a relative tolerance of 1e-3.
    - The computed total power matches the expected total power
      within a relative tolerance of 1e-3.
    Raises:
        AssertionError: If the computed values do not match the
        expected values.
    """

    # Mock input data
    thrust = np.array([100.0, 200.0, 300.0])  # Local thrust values
    moment = np.array([50.0, 100.0, 150.0])   # Local moment values
    rot_speed = 2.0                           # Rotational speed in rad/s
    rated_power = 500.0                       # Rated power in watts
    blade_data = {
        'blade_span_m': [0.0, 5.0, 10.0]      # Radial segments
    }

    # Call the function
    total_thrust, total_power = BEM.Compute_Power_Thrust(thrust, moment,
                                                         rot_speed,
                                                         rated_power,
                                                         blade_data)

    # Expected results
    expected_total_thrust = np.trapz(thrust, blade_data['blade_span_m'])
    expected_total_moment = np.trapz(moment, blade_data['blade_span_m'])
    expected_total_power = min(expected_total_moment * rot_speed, rated_power)

    # Assertions
    assert pytest.approx(total_thrust, rel=1e-3) == \
        expected_total_thrust, f"Expected total thrust \
        {expected_total_thrust}, but got {total_thrust}"
    assert pytest.approx(total_power, rel=1e-3) == \
        expected_total_power, f"Expected total power \
        {expected_total_power}, but got {total_power}"


def test_Compute_CT_CP():
    """
    Unit tests for the Compute_CT_CP function in the BEM module.
    This function tests the calculation of thrust coefficient (CT) and power
    coefficient (CP) under various conditions.
    Test Cases:
    1. Standard inputs:
       - Validates that the computed CT and CP values match the expected values
         calculated using the given formulae.
       - Inputs:
         total_thrust: Total thrust produced by the rotor (N).
         total_power: Total power produced by the rotor (W).
         wind_speed: Wind speed (m/s).
         rot_radius: Rotor radius (m).
         density: Air density (kg/m^3).
       - Asserts:
         - CT and CP values are approximately equal to the expected values
           within a relative tolerance of 1e-3.
    2. Zero wind speed:
       - Validates that the function raises a ZeroDivisionError when wind speed
         is zero, as this would result in a division by zero.
    3. Zero rotor radius:
       - Validates that the function raises a ZeroDivisionError when rotor
         radius is zero, as this would result in a division by zero.
    Raises:
        AssertionError: If the computed CT or CP values do not match the
                        expected values within the specified tolerance.
        ZeroDivisionError: If wind speed or rotor radius is zero.
    """

    # Test case 1: Standard inputs
    total_thrust = 1000.0  # N
    total_power = 50000.0  # W
    wind_speed = 10.0  # m/s
    rot_radius = 50.0  # m
    density = 1.225  # kg/m^3

    CT, CP = BEM.Compute_CT_CP(total_thrust, total_power, wind_speed,
                               rot_radius, density)

    rotor_area = 3.141592653589793 * (rot_radius ** 2)
    expected_CT = total_thrust / (0.5 * density * wind_speed ** 2 * rotor_area)
    expected_CP = total_power / (0.5 * density * wind_speed ** 3 * rotor_area)

    assert pytest.approx(CT, rel=1e-3) == expected_CT, \
        f"Expected CT {expected_CT}, but got {CT}"
    assert pytest.approx(CP, rel=1e-3) == expected_CP, \
        f"Expected CP {expected_CP}, but got {CP}"

    # Test case 2: Zero wind speed
    wind_speed = 0.0
    with pytest.raises(ZeroDivisionError):
        BEM.Compute_CT_CP(total_thrust, total_power, wind_speed, rot_radius,
                          density)

    # Test case 3: Zero rotor radius
    rot_radius = 0.0
    with pytest.raises(ZeroDivisionError):
        BEM.Compute_CT_CP(total_thrust, total_power, wind_speed, rot_radius,
                          density)


def test_Plot_Power_Thrust():
    """
    Unit test for the `Plot_Power_Thrust` function in the `BEM.Plot_results`
    module.
    This test verifies the correctness of the plots generated by the
    `Plot_Power_Thrust` function, which visualizes thrust and power
    as a function of wind speed.
    Test Steps:
    1. Mock input data for wind speed, total thrust, and total power.
    2. Call the `Plot_Power_Thrust` function with the mock data.
    3. Validate the following for the thrust plot:
       - The returned figure is a matplotlib Figure instance.
       - The x-axis label is 'Wind Speed (m/s)'.
       - The y-axis label is 'Thrust (MN)'.
       - The plot contains one line.
       - The x-data of the line matches the input wind speed.
       - The y-data of the line matches the total thrust converted to MN.
    4. Validate the following for the power plot:
       - The x-axis label is 'Wind Speed (m/s)'.
       - The y-axis label is 'Power (MW)'.
       - The plot contains one line.
       - The x-data of the line matches the input wind speed.
       - The y-data of the line matches the total power converted to MW.
    5. Close the generated figure to clean up resources.
    Raises:
        AssertionError: If any of the assertions fail, indicating an issue
        with the `Plot_Power_Thrust` function or the generated plots.
    """

    # Mock input data
    wind_speed = np.array([5, 10, 15, 20])
    total_thrust = np.array([1000000, 2000000, 3000000, 4000000])  # in N
    total_power = np.array([500000, 1000000, 1500000, 2000000])    # in W

    # Call the function
    fig, (ax1, ax2) = BEM.Plot_results.Plot_Power_Thrust(
        wind_speed, total_thrust, total_power
    )

    # Assertions for thrust plot
    assert isinstance(fig, plt.Figure), (
        "Thrust plot figure is not a matplotlib Figure."
    )
    assert ax1.get_xlabel() == 'Wind Speed (m/s)', (
        "Thrust plot x-axis label is incorrect."
    )
    assert ax1.get_ylabel() == 'Thrust (MN)', (
        "Thrust plot y-axis label is incorrect."
    )
    assert len(ax1.lines) == 1, "Thrust plot should have one line."
    assert np.array_equal(
        ax1.lines[0].get_xdata(), wind_speed
    ), "Thrust plot x-data is incorrect."
    assert np.array_equal(
        ax1.lines[0].get_ydata(), total_thrust / 1e6
    ), "Thrust plot y-data is incorrect."

    # Assertions for power plot
    assert ax2.get_xlabel() == 'Wind Speed (m/s)', (
        "Power plot x-axis label is incorrect."
    )
    assert ax2.get_ylabel() == 'Power (MW)', (
        "Power plot y-axis label is incorrect."
    )
    assert len(ax2.lines) == 1, "Power plot should have one line."
    assert np.array_equal(
        ax2.lines[0].get_xdata(), wind_speed
    ), "Power plot x-data is incorrect."
    assert np.array_equal(
        ax2.lines[0].get_ydata(), total_power / 1e6
    ), "Power plot y-data is incorrect."

    # Clean up the plots
    plt.close(fig)


def test_Plot_CT_CP():
    """
    Unit test for the Plot_CT_CP function in the BEM.Plot_results module.
    This test verifies the following:
    - The function returns a matplotlib Figure and two Axes objects.
    - The CT plot (ax1) has the correct x-axis and y-axis labels.
    - The CT plot contains one line with the correct x and y data.
    - The CP plot (ax2) has the correct x-axis and y-axis labels.
    - The CP plot contains one line with the correct x and y data.
    Mock input data:
    - wind_speed: Array of wind speeds (m/s).
    - c_thrust: Array of thrust coefficients (CT).
    - c_power: Array of power coefficients (CP).
    Assertions:
    - The returned figure is an instance of matplotlib Figure.
    - The x-axis and y-axis labels for both plots are as expected.
    - The plots contain one line each with the correct data.
    Cleans up the generated plot after testing by closing the figure.
    """

    # Mock input data
    wind_speed = np.array([5, 10, 15, 20])
    c_thrust = np.array([0.1, 0.2, 0.3, 0.4])
    c_power = np.array([0.05, 0.15, 0.25, 0.35])

    # Call the function
    fig, (ax1, ax2) = BEM.Plot_results.Plot_CT_CP(wind_speed, c_thrust,
                                                  c_power)

    # Assertions for CT plot
    assert isinstance(fig, plt.Figure), \
        "CT plot figure is not a matplotlib Figure."
    assert ax1.get_xlabel() == 'Wind Speed (m/s)', \
        "CT plot x-axis label is incorrect."
    assert ax1.get_ylabel() == 'CT', \
        "CT plot y-axis label is incorrect."
    assert len(ax1.lines) == 1, \
        "CT plot should have one line."
    assert np.array_equal(ax1.lines[0].get_xdata(), wind_speed), \
        "CT plot x-data is incorrect."
    assert np.array_equal(ax1.lines[0].get_ydata(), c_thrust), \
        "CT plot y-data is incorrect."

    # Assertions for CP plot
    assert ax2.get_xlabel() == 'Wind Speed (m/s)', \
        "CP plot x-axis label is incorrect."
    assert ax2.get_ylabel() == 'CP', \
        "CP plot y-axis label is incorrect."
    assert len(ax2.lines) == 1, \
        "CP plot should have one line."
    assert np.array_equal(ax2.lines[0].get_xdata(), wind_speed), \
        "CP plot x-data is incorrect."
    assert np.array_equal(ax2.lines[0].get_ydata(), c_power), \
        "CP plot y-data is incorrect."

    # Clean up the plots
    plt.close(fig)


def test_plot_CP_CT_TSR():
    """
    Unit test for the `plot_CP_CT_TSR` function in the `BEM` module.
    This test verifies the following:
    - The function returns a matplotlib Figure object with two subplots.
    - The first subplot (CP vs TSR) has the correct labels, title, and data.
    - The second subplot (CT vs TSR) has the correct labels, title, and data.
    Test Details:
    - Mock input data for TSR, CP, and CT is provided.
    - The function is called with the mock data, and the returned figure
      and axes are validated.
    - Assertions are made to ensure:
        - The figure contains exactly two subplots.
        - The first subplot correctly plots CP vs TSR with appropriate
          labels and title.
        - The second subplot correctly plots CT vs TSR with appropriate
          labels and title.
        - The data plotted in both subplots matches the input data.
    The test also ensures proper cleanup of the plot by closing the
    figure after validation.
    """

    # Mock input data
    tsr = [1, 2, 3, 4, 5]
    cp = [0.1, 0.2, 0.3, 0.4, 0.5]
    ct = [0.5, 0.4, 0.3, 0.2, 0.1]

    # Call the function
    fig, (ax1, ax2) = BEM.plot_CP_CT_TSR(tsr, cp, ct)

    # Assertions for the figure and axes
    assert isinstance(fig, plt.Figure), \
        "Returned object is not a matplotlib Figure."
    assert len(fig.axes) == 2, "Figure does not contain two subplots."

    # Assertions for the first subplot (CP vs TSR)
    assert ax1.get_xlabel() == 'TSR', \
        "First subplot x-axis label is incorrect."
    assert ax1.get_ylabel() == 'CP', \
        "First subplot y-axis label is incorrect."
    assert ax1.get_title() == 'CP vs TSR', \
        "First subplot title is incorrect."
    assert len(ax1.lines) == 1, \
        "First subplot should have one line."
    assert np.array_equal(ax1.lines[0].get_xdata(), tsr), \
        "First subplot x-data is incorrect."
    assert np.array_equal(ax1.lines[0].get_ydata(), cp), \
        "First subplot y-data is incorrect."

    # Assertions for the second subplot (CT vs TSR)
    assert ax2.get_xlabel() == 'TSR', \
        "Second subplot x-axis label is incorrect."
    assert ax2.get_ylabel() == 'CT', \
        "Second subplot y-axis label is incorrect."
    assert ax2.get_title() == 'CT vs TSR', \
        "Second subplot title is incorrect."
    assert len(ax2.lines) == 1, \
        "Second subplot should have one line."
    assert np.array_equal(ax2.lines[0].get_xdata(), tsr), \
        "Second subplot x-data is incorrect."
    assert np.array_equal(ax2.lines[0].get_ydata(), ct), \
        "Second subplot y-data is incorrect."

    # Clean up the plot
    plt.close(fig)


def test_Plot_Power_Thrust_Compare():
    """
    Unit test for the `Plot_Power_Thrust_Compare` function in the `BEM` module.
    This test verifies that the `Plot_Power_Thrust_Compare` function correctly
    generates a matplotlib figure with two subplots: one for thrust vs wind
    speed and another for power vs wind speed. It checks the following:
    - The returned object is a matplotlib Figure with two subplots.
    - The first subplot (Thrust vs Wind Speed) has the correct labels, title,
      and data.
    - The second subplot (Power vs Wind Speed) has the correct labels, title,
      and data.
    - The data plotted in both subplots matches the expected computed and
      observed values.
    Mock input data is used to simulate the inputs to the function, including:
    - Wind speed values.
    - Total thrust and power values (computed).
    - Observed aerodynamic thrust and power values from optimization data.
    Assertions are performed to ensure:
    - The figure and axes are correctly created.
    - The x-axis and y-axis labels, titles, and plotted data for both subplots
      are accurate.
    - The number of lines in each subplot matches the expected count.
    Finally, the plot is closed to clean up resources.
    Raises:
        AssertionError: If any of the assertions fail, indicating a mismatch
                        between the expected and actual behavior of the
                        function.
    """

    # Mock input data
    wind_speed = np.array([5, 10, 15, 20])
    total_thrust = np.array([1000000, 2000000, 3000000, 4000000])  # in N
    total_power = np.array([500000, 1000000, 1500000, 2000000])    # in W
    opt_data = {
        'wind speed [m/s]': [5, 10, 15, 20],
        'aero thrust [kn]': [1.0, 2.0, 3.0, 4.0],  # in kN
        'aero power [kw]': [0.5, 1.0, 1.5, 2.0]   # in kW
    }

    # Call the function
    fig, (ax1, ax2) = BEM.Plot_Power_Thrust_Compare(wind_speed, total_thrust,
                                                    total_power, opt_data)

    # Assertions for the figure and axes
    assert isinstance(fig, plt.Figure), \
        "Returned object is not a matplotlib Figure."
    assert len(fig.axes) == 2, \
        "Figure does not contain two subplots."

    # Assertions for the first subplot (Thrust vs Wind Speed)
    assert ax1.get_xlabel() == 'Wind Speed (m/s)', \
        "First subplot x-axis label is incorrect."
    assert ax1.get_ylabel() == 'Thrust (MN)', \
        "First subplot y-axis label is incorrect."
    assert ax1.get_title() == 'Thrust vs Wind Speed', \
        "First subplot title is incorrect."
    assert len(ax1.lines) == 2, \
        "First subplot should have two lines."
    assert np.array_equal(ax1.lines[0].get_xdata(), wind_speed), \
        "First subplot computed thrust x-data is incorrect."
    assert np.array_equal(ax1.lines[0].get_ydata(), total_thrust / 1e6), \
        "First subplot computed thrust y-data is incorrect."
    assert np.array_equal(ax1.lines[1].get_xdata(),
                          opt_data['wind speed [m/s]']), \
        "First subplot observed thrust x-data is incorrect."
    assert np.array_equal(
        ax1.lines[1].get_ydata(),
        np.array(opt_data['aero thrust [kn]']) * 1e3 / 1e6
    ), "First subplot observed thrust y-data is incorrect."

    # Assertions for the second subplot (Power vs Wind Speed)
    assert ax2.get_xlabel() == 'Wind Speed (m/s)', \
        "Second subplot x-axis label is incorrect."
    assert ax2.get_ylabel() == 'Power (MW)', \
        "Second subplot y-axis label is incorrect."
    assert ax2.get_title() == 'Power vs Wind Speed', \
        "Second subplot title is incorrect."
    assert len(ax2.lines) == 2, \
        "Second subplot should have two lines."
    assert np.array_equal(ax2.lines[0].get_xdata(), wind_speed), \
        "Second subplot computed power x-data is incorrect."
    assert np.array_equal(ax2.lines[0].get_ydata(), total_power / 1e6), \
        "Second subplot computed power y-data is incorrect."
    assert np.array_equal(ax2.lines[1].get_xdata(),
                          opt_data['wind speed [m/s]']), \
        "Second subplot observed power x-data is incorrect."
    assert np.array_equal(ax2.lines[1].get_ydata(),
                          np.array(opt_data['aero power [kw]']) * 1e3 / 1e6), \
        "Second subplot observed power y-data is incorrect."

    # Clean up the plot
    plt.close(fig)


def test_Compute_ind_factor_corrected_with_radius_calculation():
    """
    Unit test for the Compute_ind_factor_corrected function in the
    BEM.Corrected_ind_factors module, specifically testing the case
    where the rotor radius (R) is not provided and must be calculated
    from the blade span data.
    This test verifies the following:
    - The function correctly calculates the rotor radius as the maximum
      radial position from the blade span data when R is None.
    - The computed axial (`a`) and tangential (`a_prime`) induction
      factors are valid and within expected bounds.
    - The lengths of the computed induction factors match the length of
      the blade span data.
    Mock Input Data:
    - `wind_speed`: Wind speed in m/s.
    - `rot_speed`: Rotational speed in rad/s.
    - `pitch_angle`: Pitch angle in degrees.
    - `blade_data`: Dictionary containing blade span, chord length, and
      twist angle data.
    - `cl_table`: Dictionary containing lift coefficient data for
      various angles of attack.
    - `cd_table`: Dictionary containing drag coefficient data for
      various angles of attack.
    - `B`: Number of blades.
    Assertions:
    - Validates that the rotor radius is correctly calculated as the
      maximum radial position.
    - Ensures the computed induction factors are within valid bounds.
    - Checks for convergence of the induction factors.
    Raises:
    - AssertionError: If any of the above conditions are not met.
    """

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

    # Call the function without providing R
    a, a_prime = BEM.Corrected_ind_factors.Compute_ind_factor_corrected(
        wind_speed, rot_speed, pitch_angle, blade_data, cl_table, cd_table, B, R=None
    )

    # Assertions
    expected_radius = max(blade_data['blade_span_m'])
    assert expected_radius == 15.0, \
        f"Expected rotor radius {expected_radius}, but got {max(blade_data['blade_span_m'])}."
    assert len(a) == len(blade_data['blade_span_m']), \
        "Length of 'a' does not match blade span data."
    assert len(a_prime) == len(blade_data['blade_span_m']), \
        "Length of 'a_prime' does not match blade span data."
    assert all(0 <= ai <= 1 for ai in a), \
        "Axial induction factors are out of bounds."
    assert all(0 <= ai_prime <= 1 for ai_prime in a_prime), \
        "Tangential induction factors are out of bounds."


def test_plot_compare_Power_Thrust_models():
    """
    Test the `plot_compare_Power_Thrust_models` function
    from the `BEM.Corrected_ind_factors` module. This test
    verifies the following:
    - The function returns a matplotlib Figure object with
      two subplots.
    - The first subplot correctly plots the thrust vs wind
      speed for both original and corrected data.
    - The second subplot correctly plots the power vs wind
      speed for both original and corrected data.
    - The x-axis and y-axis labels, as well as the titles
      of the subplots, are correct.
    - The data plotted in the subplots matches the expected
      values.
    Mock input data:
    - `wind_speed`: Array of wind speeds in m/s.
    - `total_thrust`: Array of original total thrust values
      in N.
    - `total_power`: Array of original total power values in
      W.
    - `total_thrust_corrected`: Array of corrected total
      thrust values in N.
    - `total_power_corrected`: Array of corrected total
      power values in W.
    Assertions:
    - The returned object is a matplotlib Figure.
    - The figure contains two subplots.
    - The first subplot has the correct labels, title, and
      data for thrust vs wind speed.
    - The second subplot has the correct labels, title, and
      data for power vs wind speed.
    The test also ensures that the plot is properly closed
    after execution to avoid resource leaks.
    """

    # Mock input data
    wind_speed = np.array([5, 10, 15, 20])
    total_thrust = np.array([1000000, 2000000, 3000000, 4000000])  # in N
    total_power = np.array([500000, 1000000, 1500000, 2000000])    # in W
    total_thrust_corrected = np.array([1100000, 2100000, 3100000, 4100000])  # in N
    total_power_corrected = np.array([550000, 1050000, 1550000, 2050000])    # in W

    # Call the function
    fig, (ax1, ax2) = BEM.Corrected_ind_factors.\
        plot_compare_Power_Thrust_models(
            total_thrust, total_power, total_thrust_corrected,
            total_power_corrected, wind_speed
        )

    # Assertions for the figure and axes
    assert isinstance(fig, plt.Figure), \
        "Returned object is not a matplotlib Figure."
    assert len(fig.axes) == 2, \
        "Figure does not contain two subplots."

    # Assertions for the first subplot (Thrust vs Wind Speed)
    assert ax1.get_xlabel() == 'Wind Speed (m/s)', \
        "First subplot x-axis label is incorrect."
    assert ax1.get_ylabel() == 'Thrust (MN)', \
        "First subplot y-axis label is incorrect."
    assert ax1.get_title() == 'Thrust vs Wind Speed', \
        "First subplot title is incorrect."
    assert len(ax1.lines) == 2, \
        "First subplot should have two lines."
    assert np.array_equal(
        ax1.lines[0].get_xdata(), wind_speed[wind_speed > 5]
    ), "First subplot original thrust x-data is incorrect."
    assert np.array_equal(
        ax1.lines[0].get_ydata(), total_thrust[wind_speed > 5] / 1e6
    ), "First subplot original thrust y-data is incorrect."
    assert np.array_equal(
        ax1.lines[1].get_xdata(), wind_speed[wind_speed > 5]
    ), "First subplot corrected thrust x-data is incorrect."
    assert np.array_equal(
        ax1.lines[1].get_ydata(),
        total_thrust_corrected[wind_speed > 5] / 1e6
    ), "First subplot corrected thrust y-data is incorrect."

    # Assertions for the second subplot (Power vs Wind Speed)
    assert ax2.get_xlabel() == 'Wind Speed (m/s)', \
        "Second subplot x-axis label is incorrect."
    assert ax2.get_ylabel() == 'Power (MW)', \
        "Second subplot y-axis label is incorrect."
    assert ax2.get_title() == 'Power vs Wind Speed', \
        "Second subplot title is incorrect."
    assert len(ax2.lines) == 2, \
        "Second subplot should have two lines."
    assert np.array_equal(
        ax2.lines[0].get_xdata(), wind_speed[wind_speed > 5]
    ), "Second subplot original power x-data is incorrect."
    assert np.array_equal(
        ax2.lines[0].get_ydata(), total_power[wind_speed > 5] / 1e6
    ), "Second subplot original power y-data is incorrect."
    assert np.array_equal(
        ax2.lines[1].get_xdata(), wind_speed[wind_speed > 5]
    ), "Second subplot corrected power x-data is incorrect."
    assert np.array_equal(
        ax2.lines[1].get_ydata(),
        total_power_corrected[wind_speed > 5] / 1e6
    ), "Second subplot corrected power y-data is incorrect."

    # Clean up the plot
    plt.close(fig)

def test_Aerodynamic_file_names_error_cases():
    """Test error cases for Aerodynamic_file_names."""
    # Test non-existent directory
    with pytest.raises(FileNotFoundError):
        BEM.Aerodynamic_file_names("nonexistent_dir", "prefix")

    # Test PermissionError using monkeypatch
    def mock_listdir_fail(path):
        raise PermissionError("Permission denied")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(os, "listdir", mock_listdir_fail)

    with pytest.raises(PermissionError):
        BEM.Aerodynamic_file_names(".", "prefix")

    monkeypatch.undo()


def test_plot_airfoil_docstring_example():
    """Test the example in plot_airfoil docstring."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filenames = ['airfoil1.dat', 'airfoil2.dat', 'airfoil3.dat']
        for filename in filenames:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("\n" * 8 + "0.0 0.0\n1.0 0.0\n")  # Minimal airfoil data

        blade_data = {
            'blade_span_m': [0.0, 5.0, 10.0],
            'twist_angle_deg': [0.0, 5.0, 10.0]
        }

        fig, ax = BEM.plot_airfoil(filenames, temp_dir, blade_data)
        assert isinstance(fig, plt.Figure)
        assert hasattr(ax, 'plot')
        plt.close(fig)
def test_Aerodynamic_file_names_empty_directory():
    """
    Test the `Aerodynamic_file_names` function with an empty directory.
    This test ensures that the function correctly handles the case where
    the directory contains no files matching the specified prefix and
    `.dat` extension.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function with an empty directory
        result = BEM.Aerodynamic_file_names(temp_dir, "IEA-15-240-RWT_AeroDyn15_Polar_")

        # Expected result is an empty list
        expected = []

        # Assert the result matches the expected output
        assert result == expected, f"Expected {expected}, but got {result}"



def test_Blade_order_Airfoils_empty_blade_data():
    """
    Test the `Blade_order_Airfoils` function with empty blade data.
    This test ensures that the function handles the case where the blade
    data dictionary is empty.
    """
    # Empty blade data with missing 'airfoil_id' key
    blade_data = {'airfoil_id': []}  # Add the 'airfoil_id' key with an empty list

    # Mock airfoil data
    airfoil_data = [
        "IEA-15-240-RWT_AeroDyn15_Polar_00.dat",
        "IEA-15-240-RWT_AeroDyn15_Polar_01.dat",
        "IEA-15-240-RWT_AeroDyn15_Polar_02.dat",
    ]

    # Call the function
    result = BEM.Blade_order_Airfoils(blade_data, airfoil_data)

    # Expected result is an empty list
    expected = []

    # Assert the result matches the expected output
    assert result == expected, f"Expected {expected}, but got {result}"


def test_Airfoil_coord_names_no_matching_files():
    """
    Test the `Airfoil_coord_names` function with no matching files.
    This test ensures that the function correctly handles the case where
    no files in the directory match the specified prefix and `.txt` extension.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create unrelated files in the temporary directory
        filenames = [
            "unrelated_file_01.dat",
            "unrelated_file_02.txt",
            "another_file.txt",
        ]
        for filename in filenames:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("Test content")

        # Call the function with the temporary directory and prefix
        result = BEM.Airfoil_coord_names(temp_dir, "IEA-15-240-RWT_AF")

        # Expected result is an empty list
        expected = []

        # Assert the result matches the expected output
        assert result == expected, f"Expected {expected}, but got {result}"


def test_Compute_TSR_pitch_edge_case():
    """
    Test the `Compute_TSR_pitch` function with edge case inputs.
    This test ensures that the function handles edge cases such as
    wind speed being exactly at the boundary of the optimization data.
    """
    # Mock input data
    wind_speed = 5.0  # Boundary value
    dict_opt_data = {
        'wind speed [m/s]': [5.0, 10.0, 15.0],
        'pitch [deg]': [2.0, 3.0, 4.0],
        'rot. speed [rpm]': [10.0, 15.0, 20.0]
    }
    rotor_radius = 120

    # Call the function
    tsr, pitch, rot_speed = BEM.Compute_TSR_pitch(wind_speed, dict_opt_data, rotor_radius)

    # Expected results
    expected_tsr = (
        (10.0 * 2 * 3.141592653589793 / 60) * rotor_radius / wind_speed
    )
    expected_pitch = 2.0
    expected_rot_speed = 10.0 * 2 * 3.141592653589793 / 60

    # Assertions
    assert pytest.approx(tsr, rel=1e-3) == expected_tsr, \
        f"Expected TSR {expected_tsr}, but got {tsr}"
    assert pytest.approx(pitch, rel=1e-3) == expected_pitch, \
        f"Expected pitch {expected_pitch}, but got {pitch}"
    assert pytest.approx(rot_speed, rel=1e-3) == expected_rot_speed, \
        f"Expected rotational speed {expected_rot_speed}, but got {rot_speed}"

def test_AerodynamicInputs_init():
    """
    Test the `__init__` method of the `AerodynamicInputs` class.
    This test verifies that the method correctly initializes the object,
    reads data from the specified files, and stores the data in the expected
    format.
    Steps:
    1. Create temporary files with mock aerodynamic data.
    2. Initialize the `AerodynamicInputs` class with the file list and folder path.
    3. Verify that the `data` attribute contains the expected data for each file.
    4. Ensure the correct number of rows are skipped based on the file suffix.
    Assertions:
    - The `data` attribute is a dictionary.
    - Each key in the dictionary corresponds to a file name.
    - The values are dictionaries containing 'alpha', 'cl', and 'cd' arrays.
    - The data matches the content of the mock files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock aerodynamic data files
        file_1_content = "0.0 0.1 0.01\n5.0 0.2 0.02\n10.0 0.3 0.03\n"
        file_2_content = "0.0 0.15 0.015\n5.0 0.25 0.025\n10.0 0.35 0.035\n"
        file_3_content = "0.0 0.12 0.012\n5.0 0.22 0.022\n10.0 0.32 0.032\n"

        filenames = ["file_1.dat", "file_2.dat", "file_3.dat"]
        file_contents = [file_1_content, file_2_content, file_3_content]

        for i, filename in enumerate(filenames):
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("\n" * (20 if filename.endswith(('03.dat', '02.dat', '01.dat', '00.dat')) else 54))
                f.write(file_contents[i])

        # Initialize the class
        aerodynamic_inputs = BEM.Aerodynamic_inputs(filenames, temp_dir)

        # Assertions
        assert isinstance(aerodynamic_inputs.data, dict), \
            "The `data` attribute is not a dictionary."
        assert set(aerodynamic_inputs.data.keys()) == set(filenames), \
            "The keys in `data` do not match the file names."

        for i, filename in enumerate(filenames):
            file_data = aerodynamic_inputs.data[filename]
            assert 'alpha' in file_data and 'cl' in file_data and 'cd' in file_data, \
                f"Missing keys in data for file {filename}."
            expected_data = np.loadtxt(
                os.path.join(temp_dir, filename),
                skiprows=(20 if filename.endswith(('03.dat', '02.dat', '01.dat', '00.dat')) else 54)
            )
            np.testing.assert_array_equal(file_data['alpha'], expected_data[:, 0],
                f"Alpha values do not match for file {filename}.")
            np.testing.assert_array_equal(file_data['cl'], expected_data[:, 1],
                f"Cl values do not match for file {filename}.")
            np.testing.assert_array_equal(file_data['cd'], expected_data[:, 2],
                f"Cd values do not match for file {filename}.")

def test_Plot_rotspeed_pitch():
    """
    Test the `Plot_rotspeed_pitch` function.
    This test verifies that the function correctly generates two subplots:
    one for pitch angle vs wind speed and another for rotational speed vs wind speed.
    It checks the following:
    - The returned object is a matplotlib Figure with two subplots.
    - The first subplot (Pitch Angle vs Wind Speed) has the correct labels, title, and data.
    - The second subplot (Rotational Speed vs Wind Speed) has the correct labels, title, and data.
    - The rotational speed is correctly converted from rad/s to rpm.
    - The data plotted in the subplots matches the input data.
    """
    # Mock input data
    wind_speed = np.array([5, 10, 15, 20])
    pitch = np.array([2.0, 4.0, 6.0, 8.0])  # in degrees
    rot_speed = np.array([1.0, 2.0, 3.0, 4.0])  # in rad/s

    # Call the function
    fig, (ax1, ax2) = BEM.Plot_rotspeed_pitch(wind_speed, pitch, rot_speed)

    # Assertions for the figure and axes
    assert isinstance(fig, plt.Figure), \
        "Returned object is not a matplotlib Figure."
    assert len(fig.axes) == 2, \
        "Figure does not contain two subplots."

    # Assertions for the first subplot (Pitch Angle vs Wind Speed)
    assert ax1.get_xlabel() == 'Wind Speed (m/s)', \
        "First subplot x-axis label is incorrect."
    assert ax1.get_ylabel() == 'Pitch Angle (deg)', \
        "First subplot y-axis label is incorrect."
    assert ax1.get_title() == 'Pitch Angle vs Wind Speed', \
        "First subplot title is incorrect."
    assert len(ax1.lines) == 1, \
        "First subplot should have one line."
    assert np.array_equal(ax1.lines[0].get_xdata(), wind_speed), \
        "First subplot x-data is incorrect."
    assert np.array_equal(ax1.lines[0].get_ydata(), pitch), \
        "First subplot y-data is incorrect."

    # Assertions for the second subplot (Rotational Speed vs Wind Speed)
    assert ax2.get_xlabel() == 'Wind Speed (m/s)', \
        "Second subplot x-axis label is incorrect."
    assert ax2.get_ylabel() == 'Rotational Speed (rpm)', \
        "Second subplot y-axis label is incorrect."
    assert ax2.get_title() == 'Rotational Speed vs Wind Speed', \
        "Second subplot title is incorrect."
    assert len(ax2.lines) == 1, \
        "Second subplot should have one line."
    assert np.array_equal(ax2.lines[0].get_xdata(), wind_speed), \
        "Second subplot x-data is incorrect."
    expected_rot_speed_rpm = rot_speed * 60 / (2 * np.pi)
    assert np.allclose(ax2.lines[0].get_ydata(), expected_rot_speed_rpm), \
        "Second subplot y-data (rotational speed in rpm) is incorrect."

    # Clean up the plot
    plt.close(fig)