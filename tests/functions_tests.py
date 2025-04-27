import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib as pl
import sys
import BEM
import pandas as pd

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