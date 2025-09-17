This is a collection of scripts for running nuclear recoil simulations using
LAMMPS.

## Dependencies

These scripts require [LAMMPS](https://docs.lammps.org/Install.html) to be
installed with the KSPACE and MANYBODY packages. The
[prebuilt executables](https://docs.lammps.org/Install_linux.html) should have
these packages installed.

## Installation

Start by cloning this repository
```sh
git clone https://github.com/sebsassi/defexp.git
```
I recommend installing the scripts and running them from a Python virtual
environment. If you have the `virtualenv` package, then you can do
```sh
virtualenv venv
```
to create a virtual environment named `venv` in your working directory. Without
`virtualenv` do
```sh
python -m venv venv
```
Activate the virtual environment with
```sh
source venv/bin/activate
```
Once in the virtual environment, navigate to the directory and install the main
package with pip
```sh
cd defexp
pip install -e .
```
The flag `-e` is recommended in case you need to modify the code without
needing to reinstall the package every time.

## Usage

The main scripts to run are `relax.py` and `eloss.py`, located in the `scripts`
directory of the sources. These scripts are not installed, so you either need
to run the from the source directory
```sh
python defexp/scripts/relax.py --help
```
or just copy them to a convenient location.

The script `relax.py` needs to be run for every new material once every time
system parameters (e.g. temperature) change. It ensures that the system has the
correct lattice constants for the temperature so that the system is in a static
state for the main simulation. The script `eloss.py` is the main script for
recoil simulations.

The material and simulation parameter configurations for a couple materials are
found in the `samples` directory. Here `samples/materials` contains necessary
data to define the materials, `samples/potentials` contains the potential files
for the materials, and `samples/sim_info` contains the remaining configuration
needed to define a simulation. Generally, the material and potentials are
static information that generally doesn't need to be modified. The simulation
information, on the other hand, contains information like size of the system,
time step size, and simulation durations, which you may need to change more
regularly to tune the simulation.
