[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)


# Our Great Package

The **BEM Module** is a package based on **Blade Element Momentum (BEM) theory**, designed to help users process data, calculate aerodynamic forces, and plot wind turbine results.

# Installation Guide

The package is called 'BEM' and is already saved in this repository. To install it, open a new terminal and run the following command:

```bash
pip install -e .
```

# Colaboation

Team: [ADD TEXT HERE!]

# Overview

The code works with the wind turbine modelling project. Its purpose is to run the BEM (Blade Element Momentum) calculations to solve a wind turbine blade. The project is meant to be downloadable and then used as-is.

The output, if the code is run correctly, should be comprised of the values of the aerodynamic coefficients, forces, and powers, and the corresponding plots. Additionally, one will get a 3D visualisation of the root of the blade.

# Quick-start Guide

## Prerequisites

Before running the Blade Element Momentum (BEM) code, ensure you have the following libraries installed. The required Python version is **Python 3.11.11**.

- `numpy`
- `matplotlib`
- `pathlib`

## Input Data

The input data is provided in `main.py` and consists of the **IEA (International Energy Agency) 15 MW offshore reference turbine**.

Users should specify the following wind speed parameters based on their analysis needs:

- **Starting wind speed for analysis** (`wind_speed_start`)
- **Ending wind speed for analysis** (`wind_speed_end`)
- **Wind speed increment steps** (`wind_speed_step`)

For more details on the parameters used in the **IEA 15 MW reference wind turbine**, consult the references.

## Data provided

The dataset provided in the couse is used and this includes:

- **Blade Geometry**
- **Operational Strategy**
- **Airfoil Coordinates**
- **Aerodynamic Behaviour of Airfoils**


# Project Structure

The **Blade Element Momentum (BEM) modelling project** consists of three main code files:

- **Functions script** (`__init__.py`)
- **Tests** (`test_functions.py`)
- **Main script** (`main.py`)

## `__init__.py`

This script contains **19 functions** and **3 classes**, which are detailed in the **Architecture** section. These functions process data based on **Blade Element Momentum (BEM) theory**, calculating aerodynamic loads, power, and thrust, and generating visualizations.

### Classes:
- `Aerodynamic_inputs`
- `Corrected_ind_factors`
- `Plot_results`

### Functions:
- `Aerodynamic_file_names`
- `Read_Blade_data`
- `Blade_order_Airfoils`
- `group_data_cl`
- `group_data_cd`
- `Airfoil_coord_names`
- `plot_airfoil`
- `Blade_opt_data`
- `Compute_TSR_pitch`
- `Compute_ind_factor`
- `Compute_local_thrust_moemnt`
- `Compute_Power_Thrust`
- `Compute_CT_CP`
- `Plot_Power_Thrust`
- `Plot_CT_CP`

## `test_functions.py`

This script ensures that all functions perform correctly before they are incorporated into the main script.

## `main.py`

This script calls all classes and functions, integrating them with **IEA 15MW reference wind turbine** specifications. It also includes user-defined inputs on wind speed.


For more information regarding the structure, please look at the tree file  below. Some folders show just a few files due to the large number inside.

```text
final-project-lightning_mcteam/
├── examples/
│   ├── example.py
│   └── main.py
│
├── inputs/
│   ├── IEA-15-240-RWT/
│   │   ├── Airfoils/
│   │   │   ├── IEA-15-240-RWT_AeroDyn15_Polar_00.dat
│   │   │   ├── IEA-15-240-RWT_AeroDyn15_Polar_01.dat
│   │   │   ├── IEA-15-240-RWT_AeroDyn15_Polar_02.dat
│   │   │   └── ...
│   │   ├── IEA_15MW_RWT_Onshore.opt
│   │   └── IEA-15-240-RWT_AeroDyn15_blade.dat
│   └── rotor_diagram.jpeg
│
├── src/
│   └── BEM/
│       └── __init__.py
│
├── tests/
│   └── test_functions.py
│
├── Flow_Chart.drawio
├── LICENSE
├── pyproject.toml
└── README.md


# Architecture

Each function in the project includes a docstring explaining its purpose. Below is a summary of their functionality.

## Functions

### `Aerodynamic_file_names`
This function is used to retrieve all the files containing the aerodynamic information of different airfoils. The folder containing these files and the prefix of the file names are given as inputs to this function. It returns the file names as a list.

### `Read_Blade_data`
This function is used to read the characteristics like **sweep angle, twist angle**, etc., and the airfoil used (mentioned as a number which is later accessed by comparing the number to the name of the airfoil data) in each radial segment of the  blade. The values are stored as a dictionary.

### `Blade_order_Airfoils`
Function is designed to organize a list of aerodynamic file names (which is the aerodynamic characteristsics of different airfoils) based on the order of airfoil IDs specified in `blade_data` dictionary.

## `Aerodynamic_inputs` Class

This class is used to group the lift and drag coefficients based on the different airfoils used in different radial segments of blade and the different angles of attack (as mentioned in the aerodynamic characteristics of each airfoil). This class consists of the following functions:

### Functions in `Aerodynamic_inputs` Class:
- **`group_data_cl`** – Creates a table of lift coefficient (`Cl`) data using the angle of attack values of the first file as reference.
- **`group_data_cd`** – Creates a table of drag coefficient (`Cd`) data using the angle of attack values of the first file as reference.

## Other Functions

### `Airfoil_coord_names`
Returns the names of aerodynamic files that start with a given common prefix.

### `Plot_airfoil`
Generates a **3D plot** of the blade by aligning airfoil shapes along the span while incorporating the twist angle for each section.

### `Blade_opt_data`
Reads the **optimization blade data file** and returns a dictionary containing parameters like **span, chord length, twist angle**, and more.

### `Compute_TSR_pitch`
Calculates the **tip speed ratio (λ), pitch angle (θ), and angular velocity (ω)** for a given wind speed by interpolating θ and ω.

## `Corrected_ind_factors` Class

### `Compute_ind_factor`
Uses the **tip speed ratio (λ), pitch angle (θ), and angular velocity (ω)** to initialize aerodynamic computations.

### `Compute_local_thrust_moment`
Computes **local thrust and moment** for a wind turbine blade based on wind speed, rotational speed, and blade geometry. Iterates through blade segments and returns aerodynamic force arrays.

### `Compute_Power_Thrust`
Calculates total **thrust and power** for the turbine rotor by integrating local thrust/moment values across radial segments and applying a power limit check.

### `Compute_CT_CP`
The  function calculates the **thrust coefficient** (CT) and **power coefficient (CP)** to assess rotor performance. It uses inputs like thrust, power, wind speed, rotor radius, and air density to compute dimensionless efficiency metrics, returning CT and CP for aerodynamic analysis.

## Plotting Functions

### `Plot_Power_Thrust`
Generates two plots:
- **Thrust vs. Wind Speed**
- **Power vs. Wind Speed**

Uses Matplotlib for formatting axes, colors, and units, and returns figure and axis objects for further customization.

### `Plot_CT_CP`
Generates two plots:
- **Thrust Coefficient (CT) vs. Wind Speed**
- **Power Coefficient (CP) vs. Wind Speed**

Formats axes, colors, and grids for improved clarity and returns figure and axis objects for additional customization.

# Peer review

## Team Members

- Akheel
- Brandon
- Tudor

We all decided to meet physically and worked on one of our laptops. We tried to switch the laptops from one session to another, so that everyone gets to push/commit. All the team members have exchanged roles and have worked on all the weeks' tasks and the final project.


# References

1. Gaertner, E., Rinker, J., Sethuraman, L., Zahle, F., Anderson, B., Barter, G., Abbas, N., Meng, F., Bortolotti, P., Skrzypinski, W., Scott, G., Feil, R., Bredmose, H., Dykes, K., Shields, M., Allen, C., & Viselli, A. (2020). Definition of the IEA 15-Megawatt Offshore Reference Wind Turbine. National Renewable Energy Laboratory (NREL).