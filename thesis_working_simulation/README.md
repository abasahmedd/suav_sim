# Thesis Working Simulation

## Overview

This is an **independent working copy** of the aircraft simulation from the original [pyUAVsim](https://github.com/pabloramesc/pyUAVsim) repository, created for thesis development purposes.

## Purpose

- **Academic testing and documentation**: This copy serves as a stable baseline for running, analyzing, and documenting the UAV flight simulation as part of thesis research.
- **Later restructuring**: This working copy will be reorganized according to the thesis Chapter 4 plan in a future stage. No restructuring has been applied yet.
- **Preservation of original work**: The original repository files remain untouched. This copy preserves the original folder structure and simulation logic.

## Current State

- **Stage 1 — Working Copy**: All required files for running the simulation have been copied here with the original folder structure preserved.
- **No C++ conversion**: No code has been converted to C++ in this stage.
- **No FACE architecture**: No FACE (Future Airborne Capability Environment) restructuring has been applied.
- **No logic changes**: The simulation logic is identical to the original pyUAVsim implementation.

## How to Run

```bash
cd thesis_working_simulation
python example.py
```

### Prerequisites

Install the required Python dependencies:

```bash
pip install -r requirements.txt
pip install pymavlink
```

## Folder Structure

```
thesis_working_simulation/
├── example.py                  # Main simulation entry point
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
├── LICENSE                     # MIT License (original)
├── README.md                   # This file
├── config/                     # Aircraft parameters and waypoint files
│   ├── aerosonde_parameters.yaml
│   ├── aerosonde_parameters.json
│   ├── aerosonde_parameters.toml
│   ├── go_waypoint.wp
│   ├── example_mission.wp
│   ├── benchmark_maneuver.wp
│   └── ... (other waypoint files)
└── simulator/                  # Core simulation package
    ├── __init__.py
    ├── aircraft/               # Aircraft dynamics, aerodynamics, propulsion
    ├── autopilot/              # Autopilot, flight control, mission control, waypoints
    ├── cli/                    # Command-line interface console
    ├── common/                 # Common exceptions
    ├── environment/            # Atmospheric/environmental models
    ├── estimation/             # State estimation (EKF, filters)
    ├── gui/                    # GUI visualization panels
    ├── math/                   # Math utilities (angles, rotation, kinematics)
    ├── plot/                   # Plotting utilities
    ├── sensors/                # Sensor models
    └── utils/                  # Utility functions and data types
```

## License

This project is based on pyUAVsim, released under the **MIT License**.  
See [LICENSE](LICENSE) for the full license text.

Copyright (c) 2024 Pablo Ramirez
