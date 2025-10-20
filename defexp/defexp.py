"""
Classes for LAMMPS defect creation simulations.
"""

import os
import os.path
import sys
import shutil
import multiprocessing as mp
import logging
import time
import typing
import types
import subprocess
import json
import glob
import shutil

import numpy as np
import numpy.ma as ma
import numpy.linalg as linalg

import scipy.optimize as opt

import lammps

import ase
import ase.io.lammpsdata

from voronoi_occupation import voronoi_occupation


def ensure_file_ends_with_new_line(filename: str, verbosity: int):
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        with open(filename, "a+") as f:
            last_line = ""
            for line in f: last_line = line
            if last_line == "": f.write("\n")
            elif last_line[-1] != "\n": f.write("\n")
    else:
        with open(filename, "a") as f:
            if verbosity > 2:
                logging.debug(f"Opened file {filename} in append mode.")
            f.write("\n")


def log_print(string: str, print_: bool = True):
    logging.info(string)
    if print_: print(string)


def read_lammps_data(filename: str, verbosity: int = 0) -> ase.Atoms:
    """
    Read a simulation state from a LAMMPS data file.

    Parameters
    ----------
    filename : str
    verbosity : int

    Returns
    -------
    ase.Atoms
        Object containing the simulation state.
    """
    with open(filename, "r") as f:
        if verbosity > 1: logging.debug(f"Opened {filename} in read mode.")
        atoms = ase.io.lammpsdata.read_lammps_data(
                f, atom_style="atomic", sort_by_id=True)
        if verbosity > 1: logging.debug(f"Read data from {filename}.")
    return atoms


def is_defect_frenkel(lattice: np.ndarray, pos: np.ndarray) -> bool:
    """
    Check if perturbed lattice points have a Frenkel defect relative to the
    original lattice points.

    Parameters
    ----------
    lattice : numpy.ndarray
        Original lattice points.
    pos : numpy.ndarray
        Perturbed lattice points.

    Returns
    -------
    bool

    Notes
    -----
    Both lattice and pos must have the same shape.
    """
    return np.any(voronoi_occupation(lattice, pos) != 1)


def velocity_from(
    kinetic_energy: float, mass: float, direction: np.ndarray
) -> np.ndarray:
    """
    Produce a velocity given particle energy, mass, and a unit vector.

    Parameters
    ----------
    kinetic_energy : float
    mass : float
    direction  numpy.ndarray
        Unit vector defining the direction of velocity.

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Energy and mass must have the same units.
    """
    # Speed of light: 2997924.58 Å/ps
    return (np.sqrt(2*kinetic_energy/mass)*2997924.58)*direction


def make_atoms(
    unit_cell: np.ndarray, uc_atoms: np.ndarray, repeat: tuple, pbc: bool = True
) -> ase.Atoms:
    """
    Constructs an ASE `Atoms` object from a unit cell repeated a number of
    times along each axis.

    Parameters
    ----------
    unit_cell : numpy.ndarray
        A 3x3 array containing three vectors that define the unit cell.
    uc_atoms : numpy.ndarray
        An array containing atom positions within the unit cells in unit cell
        coordinates (i.e. each position component should be in the interval
        [0,1]).
    repeat : tuple
        Tuple of three integers defining how many times the unit cell is
        repeated in each dimension.
    pbc : bool, sequence of bools, optional
        Determines whether the boundary conditions are periodic.

    Returns
    -------
    ase.Atoms
    """
    return ase.Atoms(
            scaled_positions=uc_atoms[:,1:], numbers=uc_atoms[:,0],
            cell=unit_cell, pbc=pbc).repeat(repeat)


def uniform_angles(
    rng: np.random.Generator, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate angles from a spherically uniform distribution.

    Parameters
    ----------
    seed : int
        Seed for the RNG.
    count : int
        Number of angles to generate

    Returns
    -------
    numpy.ndarray
        Polar angles.
    numpy.ndarray
        Azimuth angles.
    """
    pa = np.arccos(2*rng.random(count) - 1)
    az = (2*np.pi)*rng.random(count)
    return pa, az


def angles_to_vec(pa: np.ndarray, az: np.ndarray) -> np.ndarray:
    """
    Transform pair of spherical angles to vector.

    Parameters
    ----------
    pa : numpy.ndarray
        Polar angle.
    az : numpy.ndarray
        Azimuthal angle.

    Returns
    -------
    numpy.ndarray
    """
    return np.stack(
            (np.sin(pa)*np.cos(az), np.sin(pa)*np.sin(az), np.cos(pa)),
            axis=-1)


def cell_integers(position: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Compute the integer coordinates of the lattice cell containing a position.

    Parameters
    ----------
    position : np.ndarray
        Position; last dimension should be 3.
    basis : np.ndarray
        3x3 array containing the primitive vectors defining the lattice.

    Returns
    -------
    np.ndarray
    """
    coordinates = np.inner(position, basis)/np.sum(basis**2, axis=1)
    botclose = np.isclose(coordinates, np.floor(coordinates), atol=1.0e-13)
    topclose = np.isclose(coordinates, np.ceil(coordinates), atol=1.0e-13)
    coordinates[botclose] = np.floor(coordinates[botclose])
    coordinates[topclose] = np.ceil(coordinates[topclose])
    return coordinates.astype(int)


class Pair:
    """
    Class for defining interatomic potential data to be read by LAMMPS.

    Attributes
    ----------
    style_name : str
        Name of the LAMMPS pair style. Refer to the LAMMPS documentation for
        options.
    pot_file : str
        Name of the potential file.
    style_args : list
        Argument list for the given style. Refer to the LAMMPS documentation
        for options.
    coeff_args : list
        Extra argument list for the pair_coeff command. Refer to the LAMMPS
        documentation for options.
    """
    def __init__(
        self, style_name: str, pot_file: str, style_args: list, coeff_args: list
    ):
        """
        Parameters
        ----------
        style_name : str
            Name of the LAMMPS pair style. Refer to the LAMMPS documentation for
            options.
        pot_file : str
            Name of the potential file.
        style_args : list
            Argument list for the given style. Refer to the LAMMPS documentation
            for options.
        coeff_args : list
            Extra argument list for the pair_coeff command. Refer to the LAMMPS
            documentation for options.
        """
        self.style_name = style_name
        self.pot_file = pot_file
        if not os.path.isfile(self.pot_file):
            raise FileNotFoundError(f"No potential file named {self.pot_file}.")

        self.style_args = style_args
        self.coeff_args = coeff_args


    def set(self, lmp: lammps.lammps):
        """
        Set pair data for a LAMMPS instance.
        """
        lmp.cmd.pair_style(self.style_name, *self.style_args)
        lmp.cmd.pair_coeff("*", "*", self.pot_file, *self.coeff_args)


class Material:
    """
    Class describing a crystalline material for LAMMPS defect simulations.

    Attributes
    ----------
    label : str
        Label for the material.
    atom_props : dict
        Dictionary where the keys are integers labeling atom types, and the
        values are dictionaries with at least the fields "mass" and "symbol"
        with the values being the mass and chemical symbol of the element.
    unit_cell : numpy.ndarray
        A 3x3 array containing three vectors that define the unit cell.
    unit_cell_atoms : numpy.ndarray
        An array containing atom positions within the unit cells in unit cell
        coordinates (i.e. each position component should be in the interval
        [0,1]).
    pair_potential : Pair
        Object that encapsulates the LAMMPS description of interatomic
        potential.
    """
    def __init__(
        self, atom_props: dict, unit_cell: np.ndarray,
        unit_cell_atoms: np.ndarray, pair_potential: Pair, 
        label: str
    ):
        """
        Parameters
        ----------
        atom_props : dict
            Dictionary where the keys are integers labeling atom types, and the
            values are dictionaries with at least the fields "mass" and "symbol"
            with the values being the mass and chemical symbol of the element.
        unit_cell : numpy.ndarray
            A 3x3 array containing three vectors that define the unit cell.
        unit_cell_atoms : numpy.ndarray
            An array containing atom positions within the unit cells in unit
            cell coordinates (i.e. each position component should be in the
            interval [0,1]).
        pair_potential : Pair
            Object that encapsulates the LAMMPS description of interatomic
            potential.
        label : str
            Label for the material.
        """
        self.label = label
        self.atom_props = atom_props
        self.unit_cell = unit_cell
        self.unit_cell_atoms = unit_cell_atoms
        self.pair_potential = pair_potential


    def set_masses(self, lmp: lammps.lammps):
        """
        Set atom masses for a LAMMPS instance.
        """
        for atom_type, props in self.atom_props.items():
            lmp.cmd.mass(atom_type, props["mass"])


class Lattice:
    """
    Class describing a lattice of atoms for LAMMPS simulations.

    Attributes
    ----------
    material : Material
        Description of the crystal structure and potential of the material.
    repeat : tuple[int]
        Number of repetitions of the unit cell along each axis.
    atoms : ase.Atoms
        Atoms forming the lattice.
    block : np.ndarray
        Box containing the atoms.
    """
    def __init__(self, material: Material, repeat: tuple[int]):
        """
        Parameters
        ----------
        material : Material
            Description of the crystal structure and potential of the material.
        repeat : tuple[int]
            Number of repetitions of the unit cell along each axis.
        """
        self.material = material
        self.repeat = repeat
        self.atoms = make_atoms(self.material.unit_cell, self.material.unit_cell_atoms, self.repeat)
        self.block = np.stack((
                repeat[0]*self.material.unit_cell[0],
                repeat[1]*self.material.unit_cell[1],
                repeat[2]*self.material.unit_cell[2]))


    @property
    def central_cell(self) -> tuple[int]:
        """
        Integer coordinates of the innermost lattice cell.
        
        Returns
        -------
        tuple[int]
        """
        return tuple(n//2 for n in self.repeat)


    def central_atom_index(self, atom_type: int, lammps: bool = False) -> int:
        """
        Index of the centermost atom of given type in the lattice.

        Parameters
        ----------
        atom_type : int
        lammps : bool, optional
            If true, gives the index in LAMMPS format (indexing starts from
            one). Otherwise indexing is assumed to start from 0.

        Returns
        -------
        int
            Index of the central atom.
        """
        avg_pos = np.mean(self.atoms.positions, axis=0)
        ds2 = np.sum((self.atoms.positions - avg_pos)**2, axis=1)
        type_mask = self.atoms.numbers != atom_type
        ds2x = ma.masked_array(ds2, type_mask)
        return np.argmin(ds2x) + 1 if lammps else np.argmin(ds2x)


    def indices_in_cell(self, cell: tuple, lammps: bool = False) -> np.ndarray:
        """
        Gives indices of the atoms contained in a given cell.

        Parameters
        ----------
        cell : tuple
            A tuple of integers specifying the cell in the lattice.
        lammps : bool, optional
            If true, gives the index in LAMMPS format (indexing starts from
            one). Otherwise indexing is assumed to start from 0.

        Returns
        -------
        numpy.ndarray
            List of indices contained in the cell.
        """
        lattice_cells = cell_integers(self.atoms.positions, self.material.unit_cell)
        mask = np.all(np.equal(lattice_cells, np.array(cell)), axis=1)
        return mask.nonzero()[0] + 1 if lammps else mask.nonzero()[0]


    def indices_in_central_cell(self, lammps: bool = False) -> np.ndarray:
        """
        Gives indices of the atoms contained in the centermost cell.

        Parameters
        ----------
        lammps : bool, optional
            If true, gives the index in LAMMPS format (indexing starts from 
            one). Otherwise indexing is assumed to start from 0.

        Returns
        -------
        numpy.ndarray
            List of indices contained in the cell.
        """
        return self.indices_in_cell(self.central_cell, lammps=False)


    def indices_in_bbox(
        self, padding: float = 0, lammps_format: bool = False
    ) -> np.ndarray:
        """
        Gives indices of atoms contained inside the lattice bounding box
        accounting for padding.

        Parameters
        ----------
        padding : float, optional
        lammps_format : bool, optional
            If true, gives the indices in LAMMPS format (indexing starts from
            one). Otherwise indexing is assumed to start from 0.

        Returns
        -------
        numpy.ndarray
            List of indices contained in the box.
        """
        max_bounds = np.max(self.block, axis=0) - padding
        min_bounds = np.min(self.block, axis=0) + padding
        max_mask = np.all(self.atoms.positions < max_bounds, axis=1)
        min_mask = np.all(self.atoms.positions > min_bounds, axis=1)
        indices = np.logical_and(max_mask, min_mask).nonzero()
        return indices + 1 if lammps_format else indices


    def interior_bbox(self, padding: float = 0) -> np.ndarray:
        """
        Gives the limits of the lattice bounding box accounting for padding.

        Parameters
        ----------
        padding : float, optional

        Returns
        -------
        numpy.ndarray
            Limits of the box.
        """
        max_bounds = np.max(self.block, axis=0) - padding
        min_bounds = np.min(self.block, axis=0) + padding
        return np.column_stack((min_bounds, max_bounds))


class ExperimentIO:
    """
    Class implementing non-LAMMPS IO operations.

    Attributes
    ----------
    label : str
        Label for the simulation.
    res_dir : str
        Directory for storing main simulation results.
    thermo_dir : str
        Directory for storing thermo output from simulation.
    log_dir : str
        Directory for storing log files.
    save_thermo : list[str], optional
        List of thermo variables to save.
    """
    def __init__(
        self, label: str, res_dir: str, thermo_dir: str, log_dir: str,
        save_thermo: typing.Optional[list[str]] = None,
    ):
        """
        Parameters
        ----------
        label : str
            Label for the simulation.
        res_dir : str
            Directory for storing main simulation results.
        thermo_dir : str
            Directory for storing thermo output from simulation.
        log_dir : str
            Directory for storing log files.
        save_thermo : list[str], optional
            List of thermo variables to save.
        """
        self.label = label

        if not os.path.isdir(res_dir):
            raise FileNotFoundError(f"{res_dir} is not a directory")
        if not os.path.isdir(thermo_dir):
            raise FileNotFoundError(f"{thermo_dir} is not a directory")
        if not os.path.isdir(log_dir):
            raise FileNotFoundError(f"{log_dir} is not a directory")

        self.res_dir = res_dir
        self.thermo_dir = thermo_dir
        self.log_dir = log_dir

        self.save_thermo = save_thermo


    def log_file_name(self, pid: int) -> str:
        return f"{self.log_dir}/{self.label}_{pid}.log"


    def output_file_name(self, *args, **kwargs) -> str:
        args_as_str = tuple(str(arg) for arg in args)
        kwargs_as_str = tuple(str(v) for k, v in kwargs if v is not None)
        fname = "_".join((self.label,) + args_as_str + kwargs_as_str) + ".dat"
        return f"{self.res_dir}/{fname}"


    def save_thermo_data(
        self, thermo_info: dict, energy: float, unitv: np.ndarray, aind: int,
        pid : int, verbosity: int = 1, binary: bool = True
    ):
        """
        save thermodynamic data marked to be saved by the save_thermo member variable.

        parameters
        ----------
        thermo_info : dict
            thermo data parsed from a lammps log file.
        energy : float
            simulated recoil energy in gev.
        unitv : np.ndarray
            simulation direction as a unit vector.
        aind : int
            index of the recoiling atom.
        pid : int
            id of the saving process.
        verbosity : bool, optional
        binary : bool, optional
            if true, save data as a .npz binary file. if false, save as a text file.
        """
        if self.save_thermo is not None:
            data = np.column_stack([thermo_info[key] for key in self.save_thermo])
            if binary:
                fname = f"{self.thermo_dir}/{self.label}_thermo_{aind}_{pid}_{hash(energy)}.dat"
                np.savez(fname,
                         energy=np.array(energy), direction=unitv,
                         columns=np.array(self.save_thermo), thermo=data)
            else:
                fname = f"{self.thermo_dir}/{self.label}_thermo_{aind}_{pid}_{hash(energy)}.npz"
                header = (
                        f"energy = {energy:.16e} ev\n"
                        f"direction = [{unitv[0]:.16e}, {unitv[1]:.16e}, "
                        f"{unitv[2]:.16e}]\n"
                        " ".join(self.save_thermo))
                np.savetxt(fname, data, header=header)
            if verbosity > 1:
                logging.debug(f"saved thermodynamic data in {fname}.")


class LAMMPSIO:
    """
    class implementing lammps io operations.

    attributes
    ----------
    label : str
        label for the simulation.
    script_dir : str
        directory containing lammps input scripts.
    work_dir : str
        directory for files used by lammps during the simulation.
    dump_dir : str
        directory for lammps dump output.
    """
    def __init__(self, label: str, work_dir: str, dump_dir: str):
        """
        class implementing lammps io operations.

        parameters
        ----------
        label : str
            label for the simulation.
        work_dir : str
            directory for files used by lammps during the simulation.
        dump_dir : str
            directory for lammps dump output.
        """
        self.label = label

        if not os.path.isdir(work_dir):
            raise filenotfounderror(f"{work_dir} is not a directory")
        if not os.path.isdir(dump_dir):
            raise filenotfounderror(f"{dump_dir} is not a directory")

        self.script_dir = f"{os.path.dirname(__file__)}/lammpsin"
        self.work_dir = work_dir
        self.dump_dir = dump_dir


    def empty_dump_dir(self):
        os.remove(glob.glob(f"{self.dump_dir}/*.dump"))
        os.remove(glob.glob(f"{self.dump_dir}/*.dump.gz"))


    def relax_script_path(self) -> str:
        return f"{self.script_dir}/relaxation.lammpsin"


    def impact_script_path(self) -> str:
        return f"{self.script_dir}/impact.lammpsin"


    def log_file_name(self, uid=None) -> str:
        if uid is None:
            return f"{self.work_dir}/{self.label}.log"
        else:
            return f"{self.work_dir}/{self.label}_{uid:d}.log"


    def pair_file_name(self, material: Material) -> str:
        return f"{self.work_dir}/{material.label}_pair.lammpsin"


    def write_pair_file(self, material: Material, verbosity: int = 0):
        """
        write a lammps input script file for the pair interaction.
        """
        fname = self.pair_file_name(material)
        with open(fname, "w") as f:
            f.truncate(0)
            style_arg_str = " ".join(str(arg) for arg in material.pair_potential.style_args)
            f.write(f"pair_style {material.pair_potential.style_name} {style_arg_str}\n")
            coeff_arg_str = " ".join(str(arg) for arg in material.pair_potential.coeff_args)
            f.write(f"pair_coeff * * \"{material.pair_potential.pot_file}\" {coeff_arg_str}\n")
        if verbosity > 1: log_print(f"wrote pair file {fname}.")


    def mass_file_name(self, material: Material) -> str:
        return f"{self.work_dir}/{material.label}_masses.lammpsin"


    def write_mass_file(self, material: Material, verbosity: int = 0):
        """
        write a file containing the atom masses in lammps script format.

        parameters
        ----------
        verbosity : int, optional
        dir : str, optional
            directory where the file is written.
        """
        fname = self.mass_file_name(material)
        with open(fname, "w") as f:
            f.truncate(0)
            for atom_type, props in material.atom_props.items():
                f.write(f"mass {atom_type:d} {props['mass']:.10f}\n")
        if verbosity > 1: log_print(f"wrote mass file {fname}.")


    def data_file_name(self, lattice: Lattice, label: str) -> str:
        return f"{self.work_dir}/{lattice.material.label}_{label}.data"


    def write_lammps_data(self, lattice: Lattice, label: str, verbosity: int = 0):
        """
        writes the lattice data into a file that can be read by lammps.

        parameters
        ----------
        verbosity : int, optional
        """
        fname = self.data_file_name(lattice, label)
        ase.io.lammpsdata.write_lammps_data(
                fname, lattice.atoms, atom_style="atomic")
        if verbosity > 1: log_print(f"wrote file {fname}.")


    def create_data_and_thermo_file(self, pid: int) -> tuple[str, str]:
        """
        creates data and thermo files if they don't already exist.

        parameters
        ----------
        pid : int
            process id.

        returns
        -------
        df_name : str
            name of output data file.
        tf_name : str
            name of thermo data file.
        """
        name = f"{self.work_dir}/{self.label}_impact_{pid}"
        df_name = f"{name}.data"
        tf_name = f"{name}.log"
        if not os.path.isfile(df_name): open(df_name,"a").close()
        if not os.path.isfile(tf_name): open(tf_name,"a").close()
        return df_name, tf_name


class RelaxSimulation:
    """
    Class defining a LAMMPS simulation used to let the lattice settle to a
    steady state at a given temperature.

    Attributes
    ----------
    lattice : Lattice
        Simulation lattice.
    lammps_io : LAMMPSIO
        Object for managing LAMMPS IO.
    verbosity : int
        Verbosity used for output.
    time_lammps : bool
        If true, the run time of the simulation is recorded.
    lammps_threads : int
        Number of threads allowed fro use by LAMMPS.
    screen : str
    timestep : float
        Timestep used in the simulation.
    temperature : float
        Temperature used in the simulation.
    num_step : int
        Number of steps the simulation should run for.
    """
    def __init__(
        self, lattice: Lattice, lammps_io: LAMMPSIO, lammps_threads: int = 1,
        screen: typing.Optional[str] = None, verbosity: int = 1,
        time_lammps: bool = False, timestep: float = 0.0002,
        duration: float = 1, temperature: float = 0.04, thermo_interval: int = 10
    ):
        """
        Parameters
        ----------
        lattice : Lattice
            Simulation lattice.
        lammps_io : LAMMPSIO
            Object for managing LAMMPS IO.
        lammps_threads : int, optional
            Number of threads allowed fro use by LAMMPS.
        screen : str, optional
        verbosity : int, optional
            Verbosity used for output.
        time_lammps : bool, optional
            If true, the run time of the simulation is recorded.
        timestep : float, optional
            Timestep used in the simulation.
        duration : float, optional
            Run time used in the simulation. The number of time steps is given
            by `int(duration/timestep)`.
        temperature : float, optional
            Temperature used in the simulation.
        """
        self.lattice = lattice
        self.thermo_interval = thermo_interval

        self.lammps_io = lammps_io
        self.lammps_io.write_lammps_data(self.lattice, "default", verbosity)
        self.lammps_io.write_mass_file(self.lattice.material, verbosity)
        self.lammps_io.write_pair_file(self.lattice.material, verbosity)

        # Logging
        self.verbosity = verbosity
        self.time_lammps = time_lammps

        # LAMMPS
        self.lammps_threads = lammps_threads
        self.screen = screen

        # LAMMPS
        self.timestep = timestep
        self.temperature = temperature
        self.num_step = int(duration/timestep)


    def run(self, uid: typing.Any = None, verbosity: int = 1):
        """
        Run the simulation.

        Parameters
        ----------
        uid : Any
            Identifier for the simulation.
        verbosity : int, optional
        """
        cmdargs = ["-log", self.lammps_io.log_file_name(uid=uid), "-echo", "both"]
        if self.screen is not None:
            cmdargs += ["-screen", self.screen]
        lmp = lammps.lammps(cmdargs=cmdargs)

        lmp.cmd.units("metal")
        lmp.cmd.atom_style("atomic")
        lmp.cmd.boundary("p", "p", "p")

        lmp.cmd.read_data(self.lammps_io.data_file_name(self.lattice, "default"))

        self.lattice.material.set_masses(lmp)
        self.lattice.material.pair_potential.set(lmp)

        lmp.cmd.timestep(self.timestep)

        lmp.cmd.neighbor(0.8, "bin")
        lmp.cmd.neigh_modify(every=10, delay=0, check=True)

        lmp.cmd.compute("EPA", "all", "pe/atom")
        lmp.cmd.compute("EKA", "all", "ke/atom")

        lmp.cmd.compute("EP", "all", "pe")

        lmp.cmd.thermo(self.thermo_interval)
        lmp.cmd.thermo_style("custom",
                "step", "time", "dt", "temp", "pe", "etotal", "press", "vol",
                "pxx", "pyy", "lx", "ly", "lz")
        lmp.cmd.thermo_modify(line="one", flush=True)
        lmp.cmd.thermo_modify("format", 1, "\"ec %8lu\"")
        lmp.cmd.thermo_modify("format", "float", "%15.10g")

        lmp.cmd.velocity("all", "create", self.temperature, 1254623,
                rot=True, mom=True, dist="gaussian")
        lmp.cmd.fix("MYNPT", "all", "npt",
                "temp", self.temperature, self.temperature, 100.0*self.timestep,
                "aniso", 0.0, 0.0, 1.0)
        lmp.cmd.run(self.num_step, post=False)
        lmp.cmd.unfix("MYNPT")

        lmp.cmd.write_data(self.lammps_io.data_file_name(self.lattice, "relaxed"))


class RecoilSimulation:
    """
    Class defining a LAMMPS simulation of defect formation.

    Attributes
    ----------
    lattice : Lattice
        Simulation lattice.
    defect_threshold : float
        Change in potential energy in eV of the system needed to flag a
        simulation as having produced defects.
    verbosity : int
        Verbosity used for output.
    time_lammps : bool
        If true, the run time of the simulation is logged.
    lammps_io : LAMMPSIO
        Object for managing LAMMPS IO.
    io : EperimentIO
        Object for managing non-LAMMPS IO.
    dump : bool
        If true, LAMMPS dump files are outputted periodically.
    lammps_threads : int
        Number of threads allowed fro use by LAMMPS.
    timestep : float
        Timestep used in the simulation.
    temperature : float
        Temperature used in the simulation.
    max_step : int
        Maximum number of steps the simulation should run for.
    relaxed_atoms : ase.Atoms
        Object containing the initial state of the lattice.
    frenkel_indices : np.ndarray
        Indices of atoms to be used in the Frenkel defect analysis.
    bbox : np.ndarray
        Bounding box of the interior region within the simulation region.
    """
    def __init__(
        self, lattice: Lattice, io: ExperimentIO, lammps_io: LAMMPSIO,
        lammps_threads: int = 1, save_thermo: typing.Optional[list] = None,
        dump: bool = False, verbosity: int = 1, time_lammps: bool = False,
        timestep: float = 0.0002, max_duration: float = 1, temperature: float = 0.04,
        border_thickness: float = 6.0, defect_threshold: float = 5,
        fit_window: float = 1.0, thermo_interval: int = 10, poterr: float = 0.5
    ):
        """
        Parameters
        ----------
        lattice : Lattice
            Simulation lattice.
        io : ExperimentIO
            Object for managing non-LAMMPS IO.
        lammps_io : LAMMPSIO
            Object for managing LAMMPS IO.
        lammps_threads : int, optional
            Number of threads allowed fro use by LAMMPS.
        dump : bool, optional
            If true, dumps intermediate simulation states.
        verbosity : int, optional
            Verbosity of logging output.
        time_lammps : bool, optional
            If true, the run time of the simulation is logged.
        timestep : float, optional
            Simulation time step. Units are determined by the units defined in
            the LAMMPS input file.
        max_duration : float, optional
            Maximum duration of the simulation. The maximum number of time
            steps the simulation runs for is given by
            `int(max_duration/timestep)`.
        temperature : float, optional
            Target temperature of the simulation.
        border_thickness : float, optional
            Thickness of the border region where temperature control is used.
            Units are determined by the units defined in the LAMMPS input file.
        defect_threshold : float, optional
            Change in potential energy in eV of the system needed to flag a 
            simulation as having produced defects.
        fit_window : float, optional
            Length of the window (in time units) used for fitting an exponential
            to the potential energy for convergence testing.
        thermo_interval : int, optional
            Number of timesteps between thermo outputs.
        poterr : float, optional
            Convergence criterion: difference between the last computed
            potential energy and the asymptotic potential energy of the fit.
        """
        self.lattice = lattice
        self.defect_threshold = defect_threshold
        self.fit_window = fit_window
        self.max_duration = max_duration
        self.poterr = poterr

        # Logging
        self.time_lammps = time_lammps

        # IO
        self.io = io
        self.lammps_io = lammps_io
        self.dump = dump
        if self.dump:
            self.lammps_io.empty_dump_dir()

        # LAMMPS
        self.lammps_threads = lammps_threads
        self.screen = None

        # LAMMPS
        self.timestep = timestep
        self.temperature = temperature
        self.thermo_interval = thermo_interval

        self.relaxed_atoms = read_lammps_data(self.lammps_io.data_file_name(self.lattice, "relaxed"))

        self.frenkel_indices = self.lattice.indices_in_bbox(
            padding=0.5, lammps_format=False)

        self.bbox = self.lattice.interior_bbox(padding=border_thickness)


    def check_for_anomalous_defect(
        self, has_frenkel_defect: bool, has_epot_defect: bool, pid: int,
        unitv: np.ndarray, energy: float, verbosity: int
    ):
        has_defect_anomaly = ((has_frenkel_defect and not has_epot_defect)
                              or (not has_frenkel_defect and has_epot_defect))

        if has_defect_anomaly:
            log_file_name = self.io.log_file_name(pid)
            if verbosity > 2:
                logging.debug(f"Opened file {logfname} in append mode.")
            with open(log_file_name, "a") as log_file:
                log_file.write(
                    f"direction = [{unitv[0]:.16e}, {unitv[1]:.16e}, "
                        f"{unitv[2]:.16e}], "
                        f"Ekin = {energy:.16e} eV\n"
                        f"Frenkel defect: {has_frenkel_defect}\n"
                        f"Epot defect: {has_epot_defect}\n\n")
            if verbosity > 2:
                logging.debug(f"Wrote to file {log_file_name}")


    def run(
        self, atom_type: int, aind: int, unitv: np.ndarray, energy: float,
        df_name: str, tf_name: str, pid: int, log_res: bool = False,
        test_frenkel: bool = True, zero_nonfrenkel: bool = True,
        seed: int = 1254623, verbosity: int = 1, adaptive_timestep: bool = True,
        max_displacement: float = 0.0005, echo: typing.Optional[str] = None,
        screen: typing.Optional[str] = None, uid: typing.Optional[int] = None
    ) -> tuple[float, bool]:
        """
        Parameters
        ----------
        atom_type : int
            Type ID of recoiling atom.
        aind : int
            Index of the recoiling atom.
        unitv : np.ndarray
            Recoil direction.
        energy : float
            Recoil energy.
        df_name : str
            Name of output data file.
        tf_name : str
            Name of thermo data file.
        pid : int
            Process ID.
        log_res : bool, optional (default: False)
            If true, prints to log file whether a defect occurred in a given
            simulation.
        test_frenkel : bool, optional
            If true, test for frenkel defects.
        zero_nonfrenkel : bool, optional
            If true, change in potential energy is zeroed if no Frenkel defects
            are detected.
        seed : int, optional
            Seed for the random number generator that determines the atom
            thermal velocities.
        verbosity : int, optional
            Verbosity of logging.
        adaptive_timestep : bool, optional
            Use an adaptive timestep in the simulation.
        max_displacement : float, optional
            Maximum distance (in simulation units) that atoms are allowed to
            move in a single time step.
        echo : str, optional
            LAMMPS echo argument: "none", "screen", "log", or "both". See
            the LAMMPS documentation for more.
        screen : str, optional
            LAMMPS screen argument: "none" or a file name. See the LAMMPS
            documentation for more.

        Returns
        -------
        depot : float
            Change in potential energy from the start of the simulation to the
            end of the simulation.
        has_frenkel_defect : bool
            Boolean of whether a Frenkel defect was detected in the simulation.
        """

        shutil.copy2(self.lammps_io.data_file_name(self.lattice, "relaxed"), df_name)

        cmdargs = ["-log", self.lammps_io.log_file_name(uid=uid), "-echo", "both"]
        if self.screen is not None:
            cmdargs += ["-screen", self.screen]
        lmp = lammps.lammps(cmdargs=cmdargs)
        lmp = lammps.lammps()

        lmp.cmd.log(self.lammps_io.log_file_name(uid=uid));

        lmp.cmd.units("metal")
        lmp.cmd.atom_style("atomic")
        lmp.cmd.boundary("p", "p", "p")

        lmp.cmd.read_data(df_name)

        self.lattice.material.set_masses(lmp)
        self.lattice.material.pair_potential.set(lmp)

        lmp.cmd.timestep(self.timestep)

        lmp.cmd.neighbor(0.8, "bin")
        lmp.cmd.neigh_modify(every=10, delay=0, check=True)

        lmp.cmd.group("RECOIL", id=aind + 1)

        lmp.cmd.region("BORDER", "block",
                self.bbox[0,0], self.bbox[0,1],
                self.bbox[1,0], self.bbox[1,1],
                self.bbox[2,0], self.bbox[2,1],
                side="out")

        lmp.cmd.region("INTERIOR", "block",
                self.bbox[0,0], self.bbox[0,1],
                self.bbox[1,0], self.bbox[1,1],
                self.bbox[2,0], self.bbox[2,1],
                side="in")

        lmp.cmd.group("BORDER_ATOMS", region="BORDER")
        lmp.cmd.group("INTERIOR_ATOMS", region="INTERIOR")

        lmp.cmd.compute("EPA", "all", "pe/atom")
        lmp.cmd.compute("EKA", "all", "ke/atom")

        lmp.cmd.compute("EP", "all", "pe")

        if self.dump:
            lmp.cmd.dump("MYDUMP", "all", "custom/gz",
                    max(self.max_step//1000, 1),
                    f"{self.lammps_io.dump_dir}/*.dump.gz",
                    "id", "type", "x", "y", "z", "c_EPA", "c_EKA",
                    "vx", "vy", "vz")
            lmp.cmd.dump_modify("MYDUMP", pad=8)

        lmp.cmd.thermo(self.thermo_interval)
        lmp.cmd.thermo_style("custom",
                "step", "time", "dt", "temp", "pe", "etotal", "press", "vol",
                "pxx", "pyy", "lx", "ly", "lz")
        lmp.cmd.thermo_modify(line="one", flush=True)
        lmp.cmd.thermo_modify("format", 1, "\"ec %8lu\"")
        lmp.cmd.thermo_modify("format", "float", "%15.10g")

        lmp.cmd.velocity("all", "create", self.temperature, seed,
                rot=True, mom=True, dist="gaussian")
        lmp.cmd.fix("MYNPT", "all", "npt",
                "temp", self.temperature, self.temperature, 100.0*self.timestep,
                "aniso", 0.0, 0.0, 1.0)
        lmp.cmd.run(20, post=False)
        lmp.cmd.unfix("MYNPT")
        thermo_info = {key: np.array([value]) for key, value in lmp.last_thermo().items()}

        if adaptive_timestep:
            lmp.cmd.fix("MYDT", "all", "dt/reset", 10, self.timestep, "NULL", max_displacement)

        vel = velocity_from(
            energy*1.0e-9,
            self.lattice.material.atom_props[atom_type]["mass"]*0.93149410242,
            unitv)
        lmp.cmd.velocity("RECOIL", "set", vel[0], vel[1], vel[2], sum=False, units="box")
        lmp.cmd.fix("CHILL", "BORDER_ATOMS", "nvt", 
                "temp", self.temperature, self.temperature, 100.0*self.timestep)
        lmp.cmd.fix("CONSERVE", "INTERIOR_ATOMS", "nve")

        def fit_func(x, a, b, c):
            return a*np.exp(-b*x) + c

        lmp.cmd.run(self.thermo_interval, post=False)
        while thermo_info["Time"][-1] < min(self.fit_window + 0.5, self.max_duration) or thermo_info["Time"].size < 200:
            lmp.cmd.run(self.thermo_interval, pre=False, post=False)

            for key, value in lmp.last_thermo().items():
                thermo_info[key] = np.append(thermo_info[key], value)

        pinit = [
            np.max(thermo_info["PotEng"]) - thermo_info["PotEng"][-1],
            1.0,
            thermo_info["PotEng"][-1]
        ]

        while (thermo_info["Time"][-1] < self.max_duration):
            lmp.cmd.run(self.thermo_interval, pre=False, post=False)

            for key, value in lmp.last_thermo().items():
                thermo_info[key] = np.append(thermo_info[key], value)

            indices = np.squeeze(np.argwhere(thermo_info["Time"][-1] - thermo_info["Time"] > self.fit_window))
            if (indices.size == 0):
                segment_start = 0
            else:
                segment_start = min(indices[-1], thermo_info["Time"].size - 200)

            time_segment = np.array(thermo_info["Time"][segment_start:])
            pot_segment = np.array(thermo_info["PotEng"][segment_start:])

            last_asymptote = pinit[2]
            try:
                popt, pcov = opt.curve_fit(fit_func, time_segment, pot_segment, p0=pinit)
                perr = np.sqrt(np.diag(pcov))
                print(perr)
                pinit = popt
            except RuntimeError:
                pinit = [
                    np.max(thermo_info["PotEng"]) - thermo_info["PotEng"][-1],
                    1.0,
                    thermo_info["PotEng"][-1]
                ]
                continue

            if np.abs(pinit[2] - pot_segment[-1]) < self.poterr and perr[2] < self.poterr:
                if thermo_info["Time"][-1] < 1.5*self.fit_window:
                    continue
                break

        positions = lmp.numpy.extract_atom("x")

        self.io.save_thermo_data(thermo_info, energy, unitv, aind, pid)

        has_frenkel_defect = False
        if test_frenkel:
            has_frenkel_defect = is_defect_frenkel(
                self.relaxed_atoms.positions[self.frenkel_indices],
                positions[self.frenkel_indices])
            if verbosity > 2:
                log_print(
                    "Tested Frenkel defect. Frenkel defects:", has_frenkel_defect)

        start_epot = thermo_info["PotEng"][0]
        end_epot = pinit[2]

        zero_condition = ((not has_frenkel_defect)
                          and test_frenkel and zero_nonfrenkel)
        depot = 0 if zero_condition else (end_epot - start_epot)

        if verbosity > 1:
            log_print(f"Checked potential energy. Difference: {depot}")

        has_epot_defect = depot > self.defect_threshold
        if log_res and test_frenkel:
            log_print(
                f"Frenkel defect: {has_frenkel_defect}\n"
                    f"Epot defect: {has_epot_defect}")

        if test_frenkel:
            self.check_for_anomalous_defect(
                has_frenkel_defect, has_epot_defect, pid, unitv, energy, verbosity)

        return depot, has_frenkel_defect


def scan_energy_range(
    recoil_simulation: RecoilSimulation, atom_type: int, aind: int,
    unitv: np.ndarray, energies: np.ndarray, pid: int, verbosity: int = 1,
    screen: typing.Optional[str] = None, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate recoils for a given direction with a range of recoil energies.

    Parameters
    ----------
    atom_type : int
        Type ID of recoiling atom.
    aind : int
        Index of the recoiling atom.
    unitv : np.ndarray
        Recoil direction.
    energies : np.ndarray
        Recoil energies.
    pid : int
        Process ID.

    Returns
    -------
    depot : np.ndarray
        Changes in potential energies from the start of the simulation to the
        end of the simulation.
    frenkel_defects : np.ndarray
        Boolean array of whether a Frenkel defect was detected in a given
        simulation.
    """
    df_name, tf_name = recoil_simulation.io.create_data_and_thermo_file(pid)

    lammps_args = {"log": tf_name}
    if screen is not None:
        lammps_args["screen"] = screen

    depot = np.zeros(energies.shape)
    frenkel_defects = np.full(energies.shape, False)
    for i in range(energies.shape[0]):
        if verbosity == 2:
            log_print(f"Energy {i + 1:d}/{energies.shape[0]:d}.")
        elif verbosity > 2:
            log_print(
                f"Energy {i + 1:d}/{energies.shape[0]:d}: "
                    f"{energies[i]:.5e}.")

        depot[i], frenkel_defects[i] = recoil_simulation.run(
            atom_type, aind, unitv, energies[i], df_name, lammps_args,
            tf_name, pid, **kwargs)

    return depot, frenkel_defects


def sample_thermal_distribution(
    recoil_simulation: RecoilSimulation, atom_type: int, aind: int,
    unitv: np.ndarray, energy: float, count: int, pid: int, seed: int = 1254623,
    screen: typing.Optional[str] = None, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate recoils for a given direction and recoil energy for multiple initial
    velocity distributions.

    Parameters
    ----------
    atom_type : int
        Type ID of recoiling atom.
    aind : int
        Index of the recoiling atom.
    unitv : np.ndarray
        Recoil direction.
    energy : float
        Recoil energy
    count : int
        Number of samples.
    pid : int
        Process ID.
    seed : int
        Seed to a random number generator.

    Returns
    -------
    depot : np.ndarray
        Changes in potential energies from the start of the simulation to the
        end of the simulation.
    frenkel_defects : np.ndarray
        Boolean array of whether a Frenkel defect was detected in a given
        simulation.
    """
    df_name, tf_name = recoil_simulation.io.create_data_and_thermo_file(pid)

    lammps_args = {"log": tf_name}
    if screen is not None:
        lammps_args["screen"] = screen

    rng = np.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(int).max, size=count)

    depot = np.zeros(count)
    frenkel_defects = np.full(count, False)
    for i in range(count):
        depot[i], frenkel_defects[i] = recoil_simulation.run(
            atom_type, aind, unitv, energy, df_name, lammps_args, tf_name,
            pid, seed=seeds[i], verbosity=verbosity, **kwargs)

    return depot, frenkel_defects


def symbols_from(atom_props: dict[str, typing.Any]) -> list[str]:
    """
    Extract list of chemical symbols from an atom property dictionary.

    Parameters
    ----------
    atom_props : dict

    Returns
    -------
    list
    """
    return [value["symbol"] for value in atom_props.values()]


def load_material(material_dir: str, potential_dir, label: str) -> Material:
    """
    Load material based on data in a config file.

    Parameters
    ----------
    material_dir : str
        Directory containing material file.
    potential_dir : str
        Directory containing potential file.
    label : str
        Label for the material.

    Returns
    -------
    Material
    """
    def parse_int_keys(x):
        return {int(k) if k.isdigit() else k: v for k, v in x}

    with open(f"{material_dir}/{label}.json", "r") as f:
        config = json.load(f, object_pairs_hook=parse_int_keys)

    return Material(
        config["atom_props"],
        np.array(config["unit_cell"]),
        np.array(config["unit_cell_atoms"]),
        Pair(
            config["pair_potential"]["style_name"],
            f"{potential_dir}/{config["pair_potential"]["pot_file"]}",
            list(config["pair_potential"]["style_args"].values()),
            symbols_from(config["atom_props"])),
        config["label"])


def load_sim_info(sim_info_dir: str, label: str) -> dict:
    with open(f"{sim_info_dir}/{label}.json", "r") as f:
        return json.load(f)
