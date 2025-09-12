import datetime
import argparse
import json
import os
import logging
import multiprocessing

import materials
import defexp

import numpy as np


def random_angle_thresholds(
    recoil_simulation: defexp.RecoilSimulation, seed: int, count: int,
    energies: np.ndarray, aind: int, pid: int, verbosity: int = 1, **kwargs
):
    """
    Run recoil simulations for multiple recoil directions.

    Parameters
    ----------
    seed : int
        Seed for a random number generator.
    count : int
        Number of directions to simulate per process.
    atom_type : int
        Type ID of recoiling atom.
    aind : int
        Index of atom used as recoiling atom.
    pid : int
        Unique process ID to distinguish simulation runs. Needed if multiple
        instances are running simultaneously to avoid overlapping use of output
        files.
    """
    rng = np.random.default_rng(seed)
    alt, az = defexp.uniform_angles(rng, count)
    unitv = defexp.angles_to_vec(alt, az)

    atom_type = recoil_simulation.lattice.atoms.numbers[aind]

    if "symbol" in recoil_simulation.material.atom_props[atom_type]:
        aid = recoil_simulation.material.atom_props[atom_type]["symbol"]
    else:
        aid = str(atom_type)

    eloss_fname = (f"{recoil_simulation.io.resdir}/"
            f"eloss_directions_{aid}_{aind}_{alt}_{az}_{pid}.dat")
    defexp.ensure_file_ends_with_new_line(eloss_fname, verbosity)

    frenkel_fname = (f"{recoil_simulation.io.resdir}/"
            f"frenkel_directions_{aid}_{aind}_{alt}_{az}_{pid}.dat")
    defexp.ensure_file_ends_with_new_line(frenkel_fname, verbosity)

    for i in range(count):
        if verbosity in (1,2):
            defexp.log_print(f"Working on direction {i + 1:d}/{count:d}.")
        elif verbosity > 2:
            defexp.log_print(
                    f"Working on direction {i + 1:d}/{count:d}: "
                    f"{alt[i]:.5f} {az[i]:.5f}.")

        depot, frenkel_defect = defexp.scan_energy_range(
                recoil_simulation, atom_type, aind, unitv[i], energies, pid, **kwargs)

        with open(eloss_fname, "a") as f:
            if verbosity > 1:
                logging.debug(f"Opened file {eloss_fname} in append mode.")
            f.write(f"{alt[i]:.16e} {az[i]:.16e} ")
            depot.tofile(f, sep=" ", format="%.16e")
            f.write("\n")

        if verbosity > 1:
            logging.debug(f"Wrote to file {eloss_fname}.")

        with open(frenkel_fname, "a") as f:
            if verbosity > 1:
                logging.debug(f"Opened file {frenkel_fname} in append mode.")
            f.write(f"{alt[i]:.16e} {az[i]:.16e} ")
            frenkel_defect.astype(int).tofile(f, sep=" ", format="%d")
            f.write("\n")

        if verbosity > 1:
            logging.debug(f"Wrote to file {frenkel_fname}.")


def execute(
    recoil_simulation: defexp.RecoilSimulation, seed: int, count: int,
    energies: np.ndarray, aind: int, pid: int, parallel: bool,
    unique_seeds: bool = True, **kwargs
):
    if parallel:
        processes = []
        for i, aind, atom_type in enumerate(zip(ainds, atom_types)):
            process_seed = abs(hash((seed, i))) if unique_seeds else seed
            args = (recoil_simulation, process_seed, count, energies, aind, i)
            processes.append(mp.Process(target=random_angle_thresholds, args=args, kwargs=kwargs))
        for p in processes: p.start()
        for p in processes: p.join()
    else:
        process_seed = abs(hash((seed, pid))) if unique_seeds else seed
        random_angle_thresholds(
                recoil_simulation, process_seed, count, energies, aind, pid, **kwargs)


if __name__ == "__main__":
    print("Running eloss.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material label")
    parser.add_argument("jid", type=int, help="job ID")
    parser.add_argument("i", type=int, help="index of recoil atom")
    parser.add_argument("seed", type=int, help="input rng seed")
    parser.add_argument("count", type=int, help="number of recoil experiments")
    parser.add_argument("-b", "--binary", type=str, default=None)
    parser.add_argument("-c", "--configpath", type=str, default=defexp.default_config_path())
    parser.add_argument("-z", "--zero-nonfrenkel", action="store_true", help="set energy loss to zero if there are no Frenkel defects")
    parser.add_argument("-p", "--parallel", action="store_true", help="run simulations in parallel")
    args = parser.parse_args()

    logging.basicConfig(
            filename=f"thresholds_{args.material}_{args.jid:d}_{args.i:d}_{args.seed:d}_{args.count:d}.log",
            level=logging.DEBUG)
    logging.info(f"Date: {datetime.date.today()}")
    logging.info(f"Material: {args.material}")
    logging.info(f"Job ID: {args.jid}")
    logging.info(f"Index: {args.i}")
    logging.info(f"Input seed: {args.seed}")
    logging.info(f"True seed: {args.seed}")
    logging.info(f"Count: {args.count}")
    logging.info(f"Config path: {args.configpath}")
    logging.info(f"Zero non-Frenkel: {args.zero_nonfrenkel}")

    if args.binary is not None:
        lammps_binary = args.binary
    else:
        lammps_binary = os.getenv("LMP_BINARY", default=None)
    if lammps_binary is None:
        raise RuntimeError(
                "Missing LAMMPS binary. Either provide name of the LAMMPS "
                "binary in the relevant command line argument or set the "
                "environment variable LMP_BINARY.")

    proj = os.getenv("PROJ", default=".")
    resdir = "/".join((proj, "thresholds", args.material))

    material = materials.load_material(f"{args.configpath}/materials", args.material)
    sim_info = defexp.load_sim_info(f"{args.configpath}/simulations", args.material)
    energies = np.loadtxt(f"{args.configpath}/energies/{args.material}.dat")

    lattice = material.lattice(sim_info["repeat"])

    exp_io = defexp.ExperimentIO(
            material.label, res_dir=resdir, save_thermo=["Time", "PotEng"])
    lammps_io = defexp.LAMMPSIO(material.label)

    simulation = defexp.RecoilSimulation(
            lammps_binary, lattice, exp_io, lammps_io, dump=False,
            time_lammps=True, timestep=sim_info["timestep"], 
            duration=sim_info["impact_duration"],
            temperature=sim_info["temperature"])

    ainds = lattice.indices_in_central_cell(lammps=False)
    atom_index = ainds[args.i % material.unit_cell_atoms.shape[0]]
    logging.info(f"Atom index: {atom_index}")

    execute(
            simulation, args.seed, args.count, energies, atom_index, args.i,
            args.parallel, unique_seeds=True, test_frenkel=True, smooth_count=10,
            zero_nonfrenkel=args.zero_nonfrenkel)
