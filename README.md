# Physical Human–Robot Interaction with a 3-DOF Delta Robot and a Custom Six-Axis Force–Torque Sensor

This repository contains the complete experimental and analytical framework used to develop and evaluate human–robot interaction behaviors on a 3-DOF Delta parallel robot equipped with a custom six-axis force–torque sensor. The work integrates hardware-dependent sensing, calibration, admittance control, Z-width passivity analysis, and experimental data collection. The included journal manuscript provides full methodological and theoretical details. 



https://github.com/user-attachments/assets/886c014f-85c3-490f-b258-71341e481e62





## Overview

The goal of this project is to characterize and improve physical human–robot interaction through real-time force sensing and admittance control. A high-precision force–torque sensor is mounted on the Delta robot end-effector, enabling accurate measurement of user-applied forces and torques. The framework includes:

- Hardware-dependent force–torque acquisition and real-time filtering  
- Full six-axis linear calibration pipeline (GUI-based)  
- Admittance and passivity-based virtual-wall simulation  
- Z-width analysis for measuring stable human–robot interaction boundaries  
- Experimental datasets used for validation and parameter identification  
- High-resolution simulation of robot kinematics and user interaction forces  

The system supports rich interaction experiments where the robot responds continuously to human input while maintaining bounded energy exchange.

## Force–Torque Sensor Calibration
<img width="792" height="448" alt="image" src="https://github.com/user-attachments/assets/fb8873a0-3010-44fc-bb80-09905cb9d4cf" />
<img width="874" height="507" alt="image" src="https://github.com/user-attachments/assets/93e088da-13bf-46bb-8ed3-81d48933e896" />

A dedicated PyQt-based tool is included for:

- Serial communication with the six-channel strain-gauge interface  
- Tare correction with 20-sample averaging  
- Per-axis linear calibration (slope and intercept estimation)  
- Real-time visualization of six channels  
- Batch calibration using known applied loads  
- Saving calibrated parameters to `slopes.npy` and `intercepts.npy`

The calibration interface implements OLS regression for each axis and allows optional filtering and zoom controls. 

## Admittance and Z-Width Simulation and Experimental Case study


A complete admittance and virtual-wall simulation environment is provided. The simulation includes:

- Full Delta-robot forward and inverse kinematics  
- Real-time admittance law with configurable mass, damping, and stiffness  
- Passivity-observer-based adaptive wall stiffness  
- Z-width estimation with peak-stiffness detection  
- Visualization of user forces, wall forces, and robot motion in 3D  
- Safety clamping and workspace boundary enforcement  

<img width="619" height="879" alt="image" src="https://github.com/user-attachments/assets/6f4e6781-9529-4f5f-a1d9-a7ac6f5a538b" />

The simulation code provides the same logic used in real experiments, enabling reproducible offline testing. 

## Experimental Data

The repository includes selected experimental datasets used for analysis:

- Raw interaction trials  
- Event-labelled sequences  
- Force, torque, and position data aligned with timestamps  

These datasets support reproduction of impedance estimation, passivity evaluation, and stability analysis.

## Code Structure

### `scripts/`
Contains simulation tools, calibration GUI, and processing utilities.

- `PHRI_SIMULATION_CODE.py` – Admittance + virtual-wall simulation with full Delta robot kinematics and force vector rendering.  
- `GUI_FT_SENSOR.py` – Six-axis calibration and monitoring tool with serial data acquisition, live plots, OLS calibration, and filtering.  
- Additional small utilities for loading calibration parameters and processing raw data.

### `data/`
Includes experimental CSV files for pHRI trials and analysis. (Hardware-dependent acquisition code is included for documentation and reproducibility.)

### `paper/`
Contains the journal manuscript describing methodology and results in detail. 

## Notes on Hardware-Dependent Components

Several components in this repository require the physical six-axis force–torque sensor and Delta robot hardware. These scripts are included for documentation, transparency, and reproducibility of the research workflow, even though they cannot be executed without the device.

## Additional Materials

> **Additional Materials**  
> Researchers who require access to extended datasets, hardware schematics, embedded firmware, or other supplementary resources associated with this work may contact **Amirhossein Ehsani** at **amirhosseinehsani80@gmail.com**. Materials can be provided upon reasonable request for scholarly and research purposes.

## License

This repository is provided for academic and research use.
