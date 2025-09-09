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

import numpy as np
import numpy.ma as ma
import numpy.linalg as linalg

import ase
import ase.io.lammpsdata

from voronoi_occupation import voronoi_occupation


def file_ends_with_newline(filename: str) -> bool:
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            for line in f: pass
        return line[-1] == "\n"
    else:
        return True


def ensure_file_ends_with_new_line(filename: str, verbosity: int):
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        with open(filename, "a+") as f:
            last_line = ""
            for line in f: last_line = line
            if last_line == "": f.write("\n")
            elif last_line[-1] != "\n": f.write("\n")
    else:
        with open(filename, "a") as f:
            if verbosity > 1:
                logging.debug(f"Opened file {filename} in append mode.")
            f.write("\n")


def log_print(string: str, print_: bool = True):
    logging.info(string)
    if print_: print(string)


def parse_log_file(log_file: typing.IO) -> dict[str, np.ndarray]:
    """
    Parse a LAMMPS log file.

    Parameters
    ----------
    log_file : file object

    Returns
    -------
    dict
        Dictionary containing thermodynamic output from the log file. Each
        key-value pair corresponds to a column of the thermodynamic output.
    """
    thermo_info = None

    log_file.seek(0)
    prev_data = []
    for line in filter(lambda x: x != "\n", log_file):
        data = line.split()
        if data[0] == "ec":
            if thermo_info is None:
                thermo_info = {column: [] for column in prev_data}
            for i, key in enumerate(thermo_info):
                type_ = int if key == "Step" else float
                thermo_info[key].append(type_(data[i + 1]))
        prev_data = data

    if thermo_info is not None:
        return {k: np.array(v) for k, v in thermo_info.items()}
    else:
        logging.error(
                f"No thermodynamic information found in file {log_file.name}")
        raise RuntimeError(
                f"No thermodynamic information found in file {log_file.name}")


def parse_log(log: str) -> dict[str, np.ndarray]:
    """
    Parse a LAMMPS log from a string.

    Parameters
    ----------
    log : str

    Returns
    -------
    dict
        Dictionary containing thermodynamic output from the log file. Each
        key-value pair corresponds to a column of the thermodynamic output.
    """
    thermo_info = None

    prev_data = []
    for line in filter(lambda x: x != "\n", log.splitlines()):
        data = line.split()
        if data[0] == "ec":
            if thermo_info is None:
                thermo_info = {column: [] for column in prev_data}
            for i, key in enumerate(thermo_info):
                type_ = int if key == "Step" else float
                thermo_info[key].append(type_(data[i + 1]))
        prev_data = data

    if thermo_info is not None:
        return {k: np.array(v) for k, v in thermo_info.items()}
    else:
        logging.error("No thermodynamic information found in log")
        raise RuntimeError("No thermodynamic information found in log")


def read_first_error_from_log_file(
        log_file: typing.IO
) -> tuple[str | None, str | None]:
    """
    Read first error message from a LAMMPS log file.

    Parameters
    ----------
    log_file : file object

    Returns
    -------
    str | NoneType
        First error message from the log file.
    str | NoneType
        Last command executed before the error.
    """
    log_file.seek(0)
    error = None
    for line in log_file:
        if line[:5] == "ERROR":
            error = line
            break
    last_command = None if error is None else log_file.readline()
    return error, last_command


def read_first_error_from_log(
        log_file: typing.IO
) -> tuple[str | None, str | None]:
    """
    Read first error message from a LAMMPS log stored in a string.

    Parameters
    ----------
    log : str

    Returns
    -------
    str | NoneType
        First error message from the log file.
    str | NoneType
        Last command executed before the error.
    """
    error = None
    for line in log.splitlines():
        if line[:5] == "ERROR":
            error = line
            break
    last_command = None if error is None else log_file.readline()
    return error, last_command


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
                f, style="atomic", sort_by_id=True)
        if verbosity > 1: logging.debug(f"Read data from {filename}.")
    return atoms


def read_thermo_info(filename: str, verbosity: int = 1) -> dict[str, np.ndarray]:
    """
    Read thermo data from a LAMMPS output file.

    Parameters
    ----------
    tf_name : str
    verbosity : int

    Returns
    -------
    dict
        Dictionary containing thermodynamic output from the log file. Each
        key-value pair corresponds to a column of the thermodynamic output.
    """
    with open(tf_name, "r") as thermo_file:
        if verbosity > 1:
            logging.debug(f"Opened file {tf_name} in append mode.")
        thermo_info = parse_log_file(thermo_file)
    if verbosity > 2:
        logging.debug(f"Read thermo info from file {tf_name}.")
    return thermo_info


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


def run_lammps(
    in_file: str, largs: dict, lvars: dict, threads: int = 1,
    binary: typing.Optional[str] = None, debug: bool = False,
    timer: bool = False, verbosity: int = 1
) -> str:
    """
    Execute a LAMMPS script.
    
    Parameters
    ----------
    in_file : str
        Name of the input file.
    largs : dict
        Dictionary of LAMMPS command line argument values with argument names
        as keys. E.g. {"a" : 3} produces "-a 3" on the command line.
    lvars : dict
        Dictionary of LAMMPS variables with variable names as keys. E.g.
        {"VAR" : "3"} produces "-v VAR 3" on the command line.
    threads : int, optional
        Number of threads to use in the simulation.
    binary : str, optional
        Name of the LAMMPS binary to use.
    debug : bool, optional
        If true, prints the command used to run LAMMPS.
    timer : bool, optional
        If true, logs the wall time it took to run LAMMPS.
    verbosity : bool, optional

    Returns
    -------
    str
        Standard output from the LAMMPS simulation.

    Notes
    -----
    To use more than one thread, the LAMMPS binary must be compiled to run
    multithreaded.
    """
    args = []

    if threads > 1: run.append(f"mpirun -np {threads:d}")

    args += [binary, f"-in \"{in_file}\""]

    if largs: run.append(" ".join(f"-{k} {v}" for k, v in largs.items()))
    if lvars: run.append(" ".join(f"-v {k} {v}" for k, v in lvars.items()))

    if verbosity > 1:
        log_print(" ".join(args))
    if timer:
        start_time = time.time()
    process = subprocess.run(args, capture_output=True, text=True)
    if timer:
        wall_elapsed = time.time() - start_time
        log_print(f"LAMMPS run took {wall_elapsed:.0f} seconds wall time")
    if process.returncode:
        with open(largs["log"]) as f:
            error, last_command = read_first_error_from_log_file(f)
        if error is not None:
            lammps_error = " ".join((error, last_command))
            error_message = (f"LAMMPS exited with exit code {process.returncode:d}. "
                    f"The following message was logged: {lammps_error}.")
        else:
            error_message = f"LAMMPS exited with exit code {process.returncode:d}."
        logging.error(error_message)
        raise RuntimeError(error_message)

    return process.stdout


def velocity_from(
    kinetic_energy: float, mass: float, unitv: np.ndarray
) -> np.ndarray:
    """
    Produce a velocity given particle energy, mass, and a unit vector.

    Parameters
    ----------
    kinetic_energy : float
    mass : float
    unitv : numpy.ndarray
        Unit vector defining the direction of velocity.

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Energy and mass must have the same units.
    """
    # Speed of light: 2997924.58 Å/ps
    return (np.sqrt(2*kinetic_energy/mass)*2997924.58)*unitv


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
        Altitude angles.
    numpy.ndarray
        Azimuth angles.
    """
    altitude = np.arccos(2*rng.random(count) - 1)
    azimuth = (2*np.pi)*rng.random(count)
    return altitude, azimuth


def angles_to_vec(alt: np.ndarray, az: np.ndarray) -> np.ndarray:
    """
    Transform pair of spherical angles to vector.

    Parameters
    ----------
    alt : numpy.ndarray
        Altitude angle.
    az : numpy.ndarray
        Azimuthal angle.

    Returns
    -------
    numpy.ndarray
    """
    return np.stack(
            (np.sin(alt)*np.cos(az), np.sin(alt)*np.sin(az), np.cos(alt)),
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


def VashishtaPair(
    pot_filename: str, atom_symbols: list, table: bool = True, N: int = 100000,
    cutoff: float = 0.2
) -> Pair:
    """
    Pair style for a Vashishta type potential.

    Parameters
    ----------
    pot_filename : str
        Name of the potential file.
    atom_symbols : list
        Chemical symbols of atoms appearing in the potential file.
    table : bool, optioonal
        If true, interpolate from tabulated potential function values to speed
        up simulations.
    N : int, optional
        Number of tabulation points to use. See LAMMPS documentation for
        recommendations.
    cutoff : float, optional
        Inner cutoff distance for the tabulation.

    Returns
    -------
    Pair
    """
    style_name = "vashishta/table" if table else "vashishta"
    style_args = (N, cutoff) if table else ()
    return Pair(style_name, pot_filename, style_args, atom_symbols)


def GWPair(pot_filename: str, atom_symbols: list) -> Pair:
    """
    Pair style for a Gao-Weber type potential.

    Parameters
    ----------
    pot_filename : str
        Name of the potential file.
    atom_symbols : list
        Chemical symbols of atoms appearing in the potential file.

    Returns
    -------
    Pair
    """
    return Pair("gw", pot_filename, (), atom_symbols)


def TersoffPair(
    pot_filename: str, atom_symbols: list, table: bool = True
) -> Pair:
    """
    Pair style for a Tersoff type potential.

    Parameters
    ----------
    pot_filename : str
        Name of the potential file.
    atom_symbols : list
        Chemical symbols of atoms appearing in the potential file.
    table : bool, optional
        If true, interpolate from tabulated potential function values to speed
        up simulations.

    Returns
    -------
    Pair
    """
    style_name = "tersoff/table" if table else "tersoff"
    return Pair(style_name, pot_filename, (), atom_symbols)


def TersoffZBLPair(pot_filename: str, atom_symbols: list) -> Pair:
    """
    Pair style for a Tersoff-ZBL type potential.

    Parameters
    ----------
    pot_filename : str
        Name of the potential file.
    atom_symbols : list
        Chemical symbols of atoms appearing in the potential file.

    Returns
    -------
    Pair
    """
    return Pair("tersoff/zbl", pot_filename, (), atom_symbols)


def EAMPair(pot_filename: str, atom_symbols: list, fs: bool = False) -> Pair:
    """
    Pair style for an EAM type potential.

    Parameters
    ----------
    pot_filename : str
        Name of the potential file.
    atom_symbols : list
        Chemical symbols of atoms appearing in the potential file.
    fs : bool, optional
        Use Finnis-Sinclair forms of potential

    Returns
    -------
    Pair
    """
    meta_data = pot_filename.split(".")
    if meta_data[-2] == "eam" and meta_data[-1] == "fs":
        return Pair("eam/fs", pot_filename, (), atom_symbols)
    elif meta_data[-1] == "eam":
        return Pair("eam", pot_filename, (), ())
    else:
        raise ValueError(
                "EAM Potential file name must end in .eam, or .eam.fs if "
                "Finnis-Sinclair potential is used.")


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
        unit_cell_atoms: np.ndarray, repeat: tuple, pair_potential: Pair, 
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


    def lattice(self, repeat: tuple[int]) -> ase.Atoms:
        return make_atoms(self.unit_cell, self.unit_cell_atoms, repeat, self.data_dir)


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
    def __init__(material: Material, repeat: tuple[int]):
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
        self.atoms = material.lattice(repeat)
        self.repeat = repeat
        self.block = np.stack((
                repeat[0]*self.material.unit_cell[0],
                repeat[1]*self.material.unit_cell[1],
                repeat[2]*self.material.unit_cell[2]))


    @property
    def central_cell(self) -> tuple[int]:
        """
        Integer coordinates of the innermost lattice cell.
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
        self, padding: float = 0, lammps: bool = False
    ) -> np.ndarray:
        """
        Gives indices of atoms contained inside the lattice bounding box
        accounting for padding.

        Parameters
        ----------
        padding : float, optional
        lammps : bool, optional
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
        return indices + 1 if lammps else indices


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
    def __init__(
        self, label: str, res_dir: str, thermo_dir: str, log_dir: str,
        save_thermo: typing.Optional[list] = None,
    ):
        self.label = experiment_label

        self.res_dir = res_dir
        self.thermo_dir = thermo_dir
        self.log_dir = log_dir

        self.save_thermo = save_thermo

        self.make_dirs()


    def make_dirs(self):
        """
        Checks if directories for output data exists and creates them if they
        do not.
        """
        if not os.path.isdir(self.res_dir): os.makedirs(self.res_dir)
        if not os.path.isdir(self.thermo_dir): os.mkdir(self.thermo_dir)
        if not os.path.isdir(self.log_dir): os.mkdir(self.log_dir)


    def log_file_name(self, pid: int):
        return f"{self.log_dir}/{self.label}_{pid}.log"


    def _create_output_fname(self, style: str, label: str, *args):
        args_as_str = tuple(str(arg) for arg in args)
        fname = "_".join((self.label, style, label) + args_as_str) + ".dat"
        return f"{self.res_dir}/{fname}"


    def eloss_fname(self, label: str, *args):
        return self._create_output_fname("eloss", label, *args)


    def frenkel_fname(self, label: str, *args):
        return self._create_output_fname("frenkel", label, *args)


    def save_thermo_data(
        self, thermo_info: dict, energy: float, unitv: np.ndarray, aind: int,
        pid : int, verbosity: int = 1
    ):
        """
        Save thermodynamic data marked to be saved by the save_thermo member variable.

        Parameters
        ----------
        thermo_info : dict
            Thermo data parsed from a LAMMPS log file.
        energy : float
            Simulated recoil energy in GeV.
        unitv : np.ndarray
            Simulation direction as a unit vector.
        aind : int
            Index of the recoiling atom.
        pid : int
            ID of the saving process.
        """
        if self.save_thermo is not None:
            data = np.column_stack([thermo_info[key] for key in self.save_thermo])
            header = (
                    f"Ekin {1.0e-9*energy:.16e} eV\n"
                    f"direction = [{unitv[0]:.16e}, {unitv[1]:.16e}, "
                    f"{unitv[2]:.16e}]\n"
                    " ".join(self.save_thermo))

            fname = f"{self.thermo_dir}/{self.label}_thermo_{aind}_{pid}_{hash(energy)}.dat"
            np.savetxt(fname, data, header=header)
            if verbosity > 1:
                logging.debug(f"Saved thermodynamic data in {fname}.")


class LAMMPSIO:
    def __init__(self, experiment_label: str, lmp_dir: str, dump_dir: str):
        self.label = experiment_label

        self.lmp_dir = lmp_dir
        self.dump_dir = dump_dir

        self.make_dirs()


    def make_dirs(self):
        """
        Checks if directories for output data exists and creates them if they
        do not.
        """
        if not os.path.isdir(self.lmp_dir): os.mkdir(self.lmp_dir)
        if self.dump_dir is not None:
            if not os.path.isdir(self.dump_dir): os.mkdir(self.dump_dir)


    def empty_dump_dir(self):
        os.system(f"rm {self.dump_dir}/*.dump {self.dump_dir}/*.dump.gz")


    def relaxation_log_fname(self, uid=None) -> str:
        if uid is None:
            return f"{self.lmp_dir}/{self.label}_relaxation.log"
        else:
            return f"{self.lmp_dir}/{self.label}_relaxation_{uid:d}.log"


    def pair_file_name(self, material: Material) -> str:
        return f"{self.lmp_dir}/{material.label}_pair.lammpsin"


    def write_pair_file(self, material: Material, verbosity: int = 0):
        """
        Write a LAMMPS input script file for the pair interaction.
        """
        fname = self.pair_file_name(material)
        with open(fname, "w") as f:
            f.truncate(0)
            style_arg_str = " ".join(str(arg) for arg in material.pair_potential.style_args)
            f.write(f"pair_style {material.pair_potential.style_name} {style_arg_str}\n")
            coeff_arg_str = " ".join(str(arg) for arg in material.pair_potential.coeff_args)
            f.write(f"pair_coeff * * \"{material.pair_potential.pot_file}\" {coeff_arg_str}\n")
        if verbosity > 1: log_print(f"Wrote pair file {fname}.")


    def mass_file_name(self, material: Material) -> str:
        return f"{self.lmp_dir}/{material.label}_masses.lammpsin"


    def write_mass_file(self, material: Material, verbosity: int = 0):
        """
        Write a file containing the atom masses in LAMMPS script format.

        Parameters
        ----------
        verbosity : int, optional
        dir : str, optional
            Directory where the file is written.
        """
        fname = self.mass_file_name(material)
        with open(fname, "w") as f:
            f.truncate(0)
            for atom_type, props in material.atom_props.items():
                f.write(f"mass {atom_type:d} {props['mass']:.10f}\n")
        if verbosity > 1: log_print(f"Wrote mass file {fname}.")


    def data_file_name(lattice: Lattice, label: str) -> str:
        return f"{self.lmp_dir}/{lattice.material.label}_{label}.data"


    def write_lammps_data(self, lattice: Lattice, data_dir: str, verbosity: int = 0):
        """
        Writes the lattice data into a file that can be read by LAMMPS.

        Parameters
        ----------
        verbosity : int, optional
        """
        fname = self.data_file_name(lattice)
        ase.io.lammpsdata.write_lammps_data(
                fname, lattice.atoms, atom_style="atomic")
        if verbosity > 1: log_print(f"Wrote file {fname}.")


    def create_data_and_thermo_file(self, pid: int) -> tuple[str, str]:
        """
        Creates data and thermo files if they don't already exist.

        Parameters
        ----------
        pid : int
            Process ID.

        Returns
        -------
        df_name : str
            Name of output data file.
        tf_name : str
            Name of thermo data file.
        """
        name = f"{self.lmp_dir}/{self.label}_impact_{pid}"
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
    material : Material
        Material used in the simulation.
    verbosity : int
        Verbosity used for output.
    time_lammps : bool
        If true, the run time of the simulation is recorded.
    io : ExperimentIO
        Object for managing simulation IO.
    lammps_threads : int
        Number of threads allowed fro use by LAMMPS.
    binary : str
        Name of the LAMMPS binary.
    mfname : str
        Name of the file containing LAMMPS readable masses of the atoms.
    pfname : str
        Name of the file describing the pair interactions for LAMMPS.
    imfname : str
        Name of the file containing the initial state of the lattice.
    screen : str
    timestep : float
        Timestep used in the simulation.
    temperature : float
        Temperature used in the simulation.
    num_step : int
        Number of steps the simulation should run for.
    config_path : str, optional
        Path to config file directory.
    """
    def __init__(
        self, binary: str, lattice: Lattice,
        lammps_io: LAMMPSIO, lammps_threads: int = 1,
        screen: typing.Optional[str] = None, verbosity: int = 1,
        time_lammps: bool = False, timestep: float = 0.0002,
        duration: float = 1, temperature: float = 0.04,
        config_path: typing.Optional[str] = None
    ):
        """
        Parameters
        ----------
        binary : str, optional
            Name of the LAMMPS binary.
        material : Material
            Material used in the simulation.
        io : ExperimentIO
            Object for managing simulation IO.
        lammps_threads : int, optional
            Number of threads allowed fro use by LAMMPS.
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
        config_path : str, optional
            Path to config file directory.
        """
        # Material
        self.material = material
        self.material.write_mass_file(verbosity)
        self.material.write_pair_file(verbosity)

        self.lattice = lattice
        self.lattice.write_lammps_data(verbosity)

        self.lammps_io = lammps_io

        # Logging
        self.verbosity = verbosity
        self.time_lammps = time_lammps

        # LAMMPS
        self.lammps_threads = lammps_threads
        self.binary = binary

        # LAMMPS
        self.screen = "none" if screen is None else screen

        # LAMMPS
        self.timestep = timestep
        self.temperature = temperature
        self.num_step = int(duration/timestep)


    def run(self, uid=None):
        """
        Run the simulation.

        Parameters
        ----------
        uid : Any
            Identifier for the simulation.
        """
        lammps_args = {
            "screen": self.screen, 
            "log": self.lammps_io.relaxation_log_fname(uid=uid)
        }

        relaxation_vars = {
            "ASTYLE": "atomic",
            "DT": self.timestep,
            "T": self.temperature,
            "STEP": self.num_step,
            "INDNAME": f"\"{self.lattice.data_file_name(lattice, "default"}\"",
            "OUTDNAME": f"\"{self.lammps_io.data_file_name(lattice, "relaxed")}\"",
            "MFNAME": f"\"{self.lammps_io.mass_filename(self.lattice.material)}\"",
            "PFNAME": f"\"{self.lammps_io.pair_filename(self.lattice.material)}\""
        }
        run_lammps(
            f"{self.lammps_io.config_path}/lammpsin/relaxation.lammpsin", lammps_args,
            relaxation_vars, threads=self.lammps_threads, binary=self.binary,
            timer=self.time_lammps)


class RecoilSimulation:
    """
    Class defining a LAMMPS simulation of defect formation.

    Attributes
    ----------
    material : Material
        Material used in the simulation.
    defect_threshold : float
        Change in potential energy in eV of the system needed to flag a
        simulation as having produced defects.
    energies : numpy.ndarray
        Simulated recoil energies in GeV.
    verbosity : int
        Verbosity used for output.
    time_lammps : bool
        If true, the run time of the simulation is logged.
    lammps_threads : int
        Number of threads allowed fro use by LAMMPS.
    binary : str
        Name of the LAMMPS binary.
    mfname : str
        Name of the file containing LAMMPS readable masses of the atoms.
    pfname : str
        Name of the file describing the pair interactions for LAMMPS.
    screen : str
        While where LAMMPS writes its screen output.
    timestep : float
        Timestep used in the simulation.
    temperature : float
        Temperature used in the simulation.
    num_step : int
        Number of steps the simulation should run for.
    relaxed_atoms : ase.Atoms
        Object containing the initial state of the lattice.
    frenkel_indices : np.ndarray
        Indices of atoms to be used in the Frenkel defect analysis.
    bbox : np.ndarray
        Bounding box of the interior region within the simulation region.
    config_path : str, optional
        Path to config file directory.
    """
    def __init__(
        self, binary: str, lattice: Lattice,
        io: ExperimentIO, lammps_io: LAMMPSIO, lammps_threads: int = 1,
        save_thermo: typing.Optional[list] = None, dump: bool = False,
        screen: typing.Optional[str] = None, verbosity: int = 1,
        time_lammps: bool = False, timestep: float = 0.0002,
        duration: float = 1, temperature: float = 0.04,
        border_thickness: float = 6.0, defect_threshold: float = 5,
        config_path: typing.Optional[str] = None
    ):
        """
        Parameters
        ----------
        binary : str
            Name of the LAMMPS binary.
        material : Material
            Material used in the simulation.
        lammps_threads : int, optional
            Number of threads allowed fro use by LAMMPS.
        io : ExperimentIO
            Object for managing simulation IO.
        dump : bool, optional
            If true, dumps intermediate simulation states.
        screen : str, optional
            While where LAMMPS writes its screen output.
        verbosity : int, optional
            Verbosity of logging output.
        time_lammps : bool, optional
            If true, the run time of the simulation is logged.
        timestep : float, optional
            Simulation time step. Units are determined by the units defined in
            the LAMMPS input file.
        duration : float, optional
            Duration of the simulation. The number of time steps the simulation
            runs is given by `int(duration/timestep)`.
        temperature : float, optional
            Target temperature of the simulation.
        border_thickness : float, optional
            Thickness of the border region where temperature control is used.
            Units are determined by the units defined in the LAMMPS input file.
        defect_threshold : float, optional
            Change in potential energy in eV of the system needed to flag a 
            simulation as having produced defects.
        config_path : str, optional
            Path to config file directory.
        """
        self.lattice = lattice
        self.defect_threshold = defect_threshold

        # Logging
        self.time_lammps = time_lammps

        # IO
        self.io = io
        self.lammps_io = lammps_io
        self._dump = dump
        if dump:
            self.io.empty_dump_dir()

        # LAMMPS
        self.lammps_threads = lammps_threads
        self.binary = binary

        # LAMMPS
        self.timestep = timestep
        self.temperature = temperature
        self.num_step = int(duration/timestep)

        self.relaxed_atoms = read_lammps_data(self.lammps_io.data_file_name(self.lattice, "relaxed"))

        self.frenkel_indices = self.lattice.indices_in_bbox(
            padding=0.5, lammps=False)

        self.bbox = self.lattice.interior_bbox(padding=border_thickness)


    def impact_vars(
        self, aind: int, atom_type: int, energy: float, unitv: np.ndarray,
        df_name: str, istep: int = 20, seed: int = 1254623
    ) -> dict:
        """
        Attach simulation arguments to LAMMPS input script variable names.

        Parameters
        ----------
        aind : int
            Index of the recoiling atom.
        atom_type : int
            Type ID of recoiling atom.
        energy : float
            Energy of the recoiling atom.
        unitv : np.ndarray
            Recoil direction.
        df_name : str
            Name of output data file.
        istep : int, optional
            Number of initial steps to run before generating the recoil.
        """
        vel = velocity_from(
            energy,
            self.material.atom_props[atom_type]["mass"]*0.93149410242,
            unitv)
        dumpint = max(self.num_step//1000, 1)
        return {
            "ASTYLE": "atomic",
            "DT": self.timestep,
            "INDNAME": f"\"{self.lammps_io.data_file_name(self.lattice, "relaxed")}\"",
            "OUTDNAME": f"\"{df_name}\"",
            "MFNAME": f"\"{self.lammps_io.mass_filename(self.lattice.material)}\"",
            "PFNAME": f"\"{self.lammps_io.pair_filename(self.lattice.material)}\"",
            "DUMP": int(self._dump),
            "DUMPINT": dumpint,
            "DUMPDIR": f"\"{self.lammps_io.dump_dir}\"",
            "T": self.temperature,
            "ISTEP": istep,
            "STEP": self.num_step,
            "CAI": aind + 1,
            "BXMIN": self.bbox[0,0], "BXMAX": self.bbox[0,1],
            "BYMIN": self.bbox[1,0], "BYMAX": self.bbox[1,1],
            "BZMIN": self.bbox[2,0], "BZMAX": self.bbox[2,1],
            "VELX": vel[0], "VELY": vel[1], "VELZ": vel[2],
            "SEED": seed
        }


    def check_for_anomalous_defect(
        has_frenkel_defect: bool, has_epot_defect: bool, pid: int,
        logfname: str, unitv: np.ndarray, energy: float, verbosity: int
    ):
        has_defect_anomaly = ((has_frenkel_defect and not has_epot_defect)
                              or (not has_frenkel_defect and has_epot_defect))

        if has_defect_anomaly and test_frenkel:
            log_file_name = self.io.log_file_name(pid)
            if verbosity > 2:
                logging.debug(f"Opened file {logfname} in append mode.")
            with open(log_file_name, "a") as log_file:
                log_file.write(
                    f"direction = [{unitv[0]:.16e}, {unitv[1]:.16e}, "
                        f"{unitv[2]:.16e}], "
                        f"Ekin = {energy:.16e} GeV\n"
                        f"Frenkel defect: {has_frenkel_defect}\n"
                        f"Epot defect: {has_epot_defect}\n\n")
            if verbosity > 2:
                logging.debug(f"Wrote to file {log_file_name}")


    def run(
        self, atom_type: int, aind: int, unitv: np.ndarray, energy: float,
        df_name: str, lammps_args: dict, tf_name: str, pid: int,
        log_res: bool = False, test_frenkel: bool = True,
        zero_nonfrenkel: bool = True, smooth_count: int = 1,
        seed: int = 1254623, verbosity: int = 1
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
        lammps_args : str
            Command line arguments to LAMMPS executable.
        tf_name : str
            Name of thermo data file.
        pid : int
            Process ID.
        log_res : bool, optional
            If true, prints to log file whether a defect occurred in a given
            simulation.
        test_frenkel : bool, optional
            If true, test for frenkel defects.
        zero_nonfrenkel : bool, optional
            If true, change in potential energy is zeroed if no Frenkel defects
            are detected.
        smooth_count : int, optional
            The final potential energy value used to calculate the change in
            energy is the mean potential energy of `smooth_count` potential
            energy values.
        seed : int, optional
            Seed for the random number generator that determines the atom
            thermal velocities.

        Returns
        -------
        dEpot : float
            Change in potential energy from the start of the simulation to the
            end of the simulation.
        has_frenkel_defect : bool
            Boolean of whether a Frenkel defect was detected in the simulation.
        """
        impact_vars = self.impact_vars(
            aind, atom_type, energy, unitv, df_name, istep=20, seed=seed)
        run_lammps(
            f"{self.config_path}/lammpsin/impact.lammpsin", lammps_args, 
            impact_vars, self.lammps_threads, binary=self.binary,
            timer=self.time_lammps)

        impacted_atoms = read_lammps_data(df_name)
        thermo_info = read_thermo_info(tf_name)

        self.io.save_thermo_data(thermo_info, energy, unitv, aind, pid)

        has_frenkel_defect = False
        if test_frenkel:
            has_frenkel_defect = is_defect_frenkel(
                self.relaxed_atoms.positions[self.frenkel_indices],
                impacted_atoms.positions[self.frenkel_indices])
            if verbosity > 2:
                log_print(
                    "Tested Frenkel defect. Frenkel defects:", has_frenkel_defect)

        epot = thermo_info["PotEng"]

        end_epot = np.mean(epot[-smooth_count:])
        zero_condition = ((not has_frenkel_defect)
                          and test_frenkel and zero_nonfrenkel)
        depot = 0 if zero_condition else (end_epot - epot[0])

        if verbosity > 2:
            log_print(f"Checked potential energy. Difference: {depot}")

        has_epot_defect = dEpot > self.defect_threshold
        if log_res and test_frenkel:
            log_print(
                f"Frenkel defect: {has_frenkel_defect}\n"
                    f"Epot defect: {has_epot_defect}")

        if test_frenkel:
            self.check_for_anomalous_defect(
                has_frenkel_defect, has_epot_defect, pid, logfname, unitv,
                energy, verbosity)

        return depot, has_frenkel_defect


def scan_energy_range(
    recoil_simulation: defexp.RecoilSimulation, atom_type: int, aind: int,
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
    dEpot : np.ndarray
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

    dEpot = np.zeros(energies.shape)
    frenkel_defects = np.full(energies.shape, False)
    for i in range(energies.shape[0]):
        if verbosity == 2:
            defexp.log_print(f"Energy {i + 1:d}/{energies.shape[0]:d}.")
        elif verbosity > 2:
            defexp.log_print(
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
    dEpot : np.ndarray
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
            pid, seed=seeds[i], verbosity, **kwargs)

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


def load_material(dir: str, label: str) -> defexp.Material:
    """
    Load material based on data in a config file.

    Parameters
    ----------
    dir : str
        Materials directory.
    label : str
        Name of material.

    Returns
    -------
    Material
    """
    def parse_int_keys(x):
        return {int(k) if k.isdigit() else k: v for k, v in x}

    with open(f"{dir}/{label}.json", "r") as f:
        config = json.load(f, object_pairs_hook=parse_int_keys)

    return defexp.Material(
        config["atom_props"],
        config["unit_cell"],
        config["unit_cell_atoms"],
        Pair(
            config["pair_potential"]["style_name"],
            config["pair_potential"]["pot_file"],
            list(config["pair_potential"]["style_args"].values()),
            symbols_from(config["atom_props"])),
        config["label"])


def load_sim_info(dir: str, label: st) -> dict:
    with open(f"{dir}/{label}.json", "r") as f:
        return json.load(f)
