import time
import datetime
import argparse
import json
import os
import logging
import typing

import numpy as np

import defexp


def random_directions(rng, central_dir: tuple[float], max_dev: float, count: int):
    sinpa = np.sin(central_dir[0])
    cospa = np.cos(central_dir[0])
    sinaz = np.sin(central_dir[1])
    cosaz = np.cos(central_dir[1])

    z = np.array([sinpa*cosaz, sinpa*sinaz, cospa])
    y = np.array([cospa*cosaz, cospa*sinaz, -sinpa])
    x = np.array([sinaz, -cosaz, 0.0])

    cosmin = np.cos(max_dev)
    umin = 0.5*(1.0 + cosmin)

    cosdev = cosmin + 2.0*(1.0 - umin)*rng.random(count)
    sindev = np.sqrt((1.0 - cosdev)*(1.0 + cosdev))
    rot = 2.0*np.pi*rng.random(count)
    cosrot = np.cos(rot)
    sinrot = np.sin(rot)

    dir = z*cosdev + y*sindev*sinrot + x*sindev*cosrot
    pa = np.arccos(dir[2])
    az = np.atan2(dir[1], dir[0])

    return pa, az, dir


def generate_directions(rng, direction, max_angle, count):

    if direction[0] == 0.0 and direction[1] == 0.0 and max_angle == np.pi:
        pa, az = defexp.uniform_angles(rng, count)
        unitv = defexp.angles_to_vec(pa, az)
    elif max_angle == 0.0:
        pa = direction[0]*np.ones(count)
        az = direction[1]*np.ones(count)
        unitv = defexp.angles_to_vec(pa, az)
    else:
        pa, az, unitv = random_directions(rng, direction, max_angle, count)

    return pa, az, unitv


def generate_energies(rng, emin, emax, count):
    if emin >= emax:
        return emin*np.ones(count)
    else:
        return emin + (emax - emin)*rng.random(count)


def generate_atom_indices(rng, ainds, count):
    return rng.choice(ainds, count)


def random_energy_loss(
    recoil_simulation: defexp.RecoilSimulation, seed: int, count: int,
    emin: float, emax: float, pid: int, pid_to_index: bool = False,
    direction: tuple[float] = (0.0, 0.0), max_angle: float = np.pi,
    verbosity: int = 1, **kwargs
):
    rng = np.random.default_rng(seed)
    pa, az, unitv = generate_directions(rng, direction, max_angle, count)
    energies = generate_energies(rng, emin, emax, count)

    ainds = recoil_simulation.lattice.indices_in_central_cell(lammps=False)
    if pid_to_index:
        ainds = lattice.indices_in_central_cell(lammps=False)
        atom_index = ainds[pid % recoil_simulation.lattice.material.unit_cell_atoms.shape[0]]
        atom_type = recoil_simulation.lattice.atoms.numbers[atom_index]

        symbol = recoil_simulation.lattice.material.atom_props[atom_type]["symbol"]

        result_fname = recoil_simulation.io.output_file_name(symbol, atom_index, pid)
        defexp.ensure_file_ends_with_new_line(result_fname, verbosity)

        for i in range(count):
            if verbosity > 1:
                defexp.log_print(
                        f"Working on sample {i + 1:d}/{count:d}: "
                        f"{pa[i]:.5f} {az[i]:.5f} {energies[i]:.5e}.")
            elif verbosity == 1:
                defexp.log_print(f"Working on sample {i + 1:d}/{count:d}.")

            depot, frenkel_defect = recoil_simulation.run(
                    atom_type, atom_index, unitv[i], energies[i], pid,
                    verbosity=verbosity, **kwargs)

            with open(result_fname, "a") as f:
                if verbosity > 2:
                    logging.debug(f"Opened file {result_fname} in append mode.")
                f.write(
                        f"{pa[i]:.16e} {az[i]:.16e} {energies[i]:.16e} "
                        f"{depot:.16e} {int(frenkel_defect)}\n")

            if verbosity > 2:
                logging.debug(f"Wrote to file {result_fname}.")
    else:
        result_fnames = {
            atom_type: recoil_simulation.io.output_file_name(prop["symbol"], "random_atom", pid)
            for atom_type, prop in recoil_simulation.lattice.material.atom_props.items()
        }
        defexp.ensure_file_ends_with_new_line(result_fname, verbosity)
        atom_indices = generate_atom_indices(ainds, count)

        for i in range(count):
            if verbosity > 1:
                defexp.log_print(
                        f"Working on sample {i + 1:d}/{count:d}: "
                        f"{atom_indices[i]} {pa[i]:.5f} {az[i]:.5f} {energies[i]:.5e}.")
            elif verbosity == 1:
                defexp.log_print(f"Working on sample {i + 1:d}/{count:d}.")

            atom_type = recoil_simulation.lattice.atoms.numbers[atom_indices[i]]
            depot, frenkel_defect = recoil_simulation.run(
                    atom_type, atom_indices[i], unitv[i], energies[i], pid,
                    verbosity=verbosity, **kwargs)

            result_fname = result_fnames[atom_type]
            with open(result_fname, "a") as f:
                if verbosity > 2:
                    logging.debug(f"Opened file {result_fname} in append mode.")
                f.write(
                        f"{atom_indices[i]} {pa[i]:.16e} {az[i]:.16e} {energies[i]:.16e} "
                        f"{depot:.16e} {int(frenkel_defect)}\n")

            if verbosity > 2:
                logging.debug(f"Wrote to file {result_fname}.")



def execute(
    recoil_simulation: defexp.RecoilSimulation, seed: int, count: int,
    emin: float, emax: float, pid: int, **kwargs
):
    random_energy_loss(
            recoil_simulation, seed, count, emin, emax, pid, **kwargs)


def angle_pair(arg: str):
    pair = arg.split(",")
    return float(pair[0]), float(pair[1])

def str_list(arg: str):
    return arg.split(",")

if __name__ == "__main__":
    print("Running eloss.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material name")
    parser.add_argument("jid", type=int, help="job ID")
    parser.add_argument("pid", type=int, help="process ID")
    parser.add_argument("seed", type=int, help="input rng seed")
    parser.add_argument("count", type=int, help="number of recoil experiments")
    parser.add_argument("-C", "--config-dir", type=str, default=".", help="directory containing material/simulation configuration files")
    parser.add_argument("-c", "--constant-timestep", action="store_true", help="do not use adaptive timestep")
    parser.add_argument("-D", "--direction", type=float, nargs=2, default=[0.0, 0.0], help="recoil directon as an angle pair ALT AZ in radians")
    parser.add_argument("-d", "--dump", action="store_true", help="make periodic dumps of simulation state")
    parser.add_argument("-E", "--energy", type=float, default=None, help="fixed recoil energy")
    parser.add_argument(      "--emin", type=float, default=None, help="minimum recoil energy")
    parser.add_argument(      "--emax", type=float, default=None, help="maximum recoil energy")
    parser.add_argument(      "--extra-label", type=str, default=None, help="extra label to attach to file names")
    parser.add_argument("-I", "--input-file", type=str, default=None, help="JSON file providing same parameters as the command line (command line arguments override values in the file)")
    parser.add_argument("-a", "--max-angle", type=float, default=np.pi, help="maximum deviation from the average recoil direction")
    parser.add_argument(      "--max-displacement", type=float, default=None, help="maximum atom displacement allowed in a single timestep")
    parser.add_argument(      "--max-duration", type=float, default=None, help="maximum simulation duration in picoseconds")
    parser.add_argument("-p", "--pid-to-index", action="store_true", help="use process ID to index into unit cell; otherwise sample randomly")
    parser.add_argument("-r", "--raw-seed", action="store_true", help="use seed as is without mixing with jid, i, and timestamp")
    parser.add_argument(      "--repeat", type=float, nargs=3, default=None, help="number of repeated unit cells along each axis")
    parser.add_argument("-R", "--res-dir", type=str, default=".", help="output directory for main results")
    parser.add_argument("-s", "--screen", action="store_true", help="print LAMMPS output to screen")
    parser.add_argument(      "--temperature", type=float, default=None, help="temperature of the system")
    parser.add_argument(      "--thermo", type=str, nargs="+", default=["Time","PotEng"], help="list of thermo quantities to save")
    parser.add_argument("-t", "--timeless-seed", action="store_true", help="do not mix timestamp into seed")
    parser.add_argument("-T", "--timestep", type=float, default=None, help="minimum simulation timestep in picoseconds")
    parser.add_argument("-W", "--work-dir", type=str, default=".", help="output directory for intermediate/auxillary files")
    parser.add_argument("-v", "--verbosity", type=int, default=2, help="verbosity of output")
    parser.add_argument("-z", "--zero-nonfrenkel", action="store_true", help="set energy loss to zero if there are no Frenkel defects")
    args = parser.parse_args()

    if args.input_file is not None:
        with open(args.input_file, "r") as f:
            arguments = json.load(f)

        for key, value in arguments.items():
            if key in vars(args).keys():
                if getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)

    if args.timestep is None:
        raise RuntimeError("Argument `timestep` needs to be defined either in an input file or via the command line.")

    timestamp = int(time.time())
    if (args.raw_seed):
        seed = args.seed
    elif args.timeless_seed:
        seed = abs(hash((args.seed, args.jid, args.pid)))
    else:
        seed = abs(hash((args.seed, args.jid, args.pid, timestamp)))

    if args.extra_label is None:
        logging.basicConfig(
                filename=f"eloss_{args.material}_{args.jid:d}_{args.pid:d}_{args.seed:d}_{args.count:d}.log",
                level=logging.DEBUG)
    else:
        logging.basicConfig(
                filename=f"eloss_{args.material}_{args.extra_label}_{args.jid:d}_{args.pid:d}_{args.seed:d}_{args.count:d}.log",
                level=logging.DEBUG)
    logging.info(f"Date: {datetime.datetime.fromtimestamp(timestamp)}")
    logging.info(f"Material: {args.material}")
    logging.info(f"Job ID: {args.jid}")
    logging.info(f"Process ID: {args.pid}")
    logging.info(f"Input seed: {args.seed}")
    logging.info(f"True seed: {seed}")
    logging.info(f"Count: {args.count}")
    logging.info(f"Config path: {args.config_dir}")
    logging.info(f"Result directory: {args.res_dir}")
    logging.info(f"Work directory: {args.work_dir}")
    logging.info(f"Zero non-Frenkel: {args.zero_nonfrenkel}")
    logging.info(f"Direction: {args.direction}")
    logging.info(f"Maximum angle: {args.max_angle}")
    logging.info(f"Energy: {args.energy}")
    logging.info(f"Minimum energy: {args.emin}")
    logging.info(f"Maximum energy: {args.emax}")
    logging.info(f"Maximum duration: {args.max_duration}")
    logging.info(f"Index: {args.index}")
    logging.info(f"Timestep: {args.timestep}")
    logging.info(f"Constant timestep {args.constant_timestep}")
    logging.info(f"Dump: {args.dump}")
    logging.info(f"Timeless seed: {args.timeless_seed}")

    res_dir = f"{args.res_dir}/eloss/{args.material}"
    lmp_dir = f"{args.work_dir}/lammps_work"
    dump_dir = f"{args.work_dir}/dump"
    thermo_dir = f"{args.work_dir}/thermo"
    log_dir = f"{args.work_dir}/logs"

    material = defexp.load_material(f"{args.config_dir}/materials", f"{args.config_dir}/potentials", args.material)

    lattice = defexp.Lattice(material, args.repeat)

    if args.energy is not None:
        emin = args.energy
        emax = args.energy

    if args.extra_label is None:
        label = f"eloss_{material.label}"
    else:
        label = f"eloss_{material.label}_{args.extra_label}"
    exp_io = defexp.ExperimentIO(
            label, res_dir, thermo_dir, log_dir, save_thermo=args.thermo)
    lammps_io = defexp.LAMMPSIO(label, lmp_dir, dump_dir)

    screen = None if args.screen else "none"
    simulation = defexp.RecoilSimulation(
            lattice, exp_io, lammps_io, dump=args.dump, time_lammps=True,
            timestep=args.timestep, max_duration=args.max_duration, temperature=args.temperature,
            screen=screen, verbosity=args.verbosity)

    random_energy_loss(
            simulation, seed, args.count, emin, emax, args.pid,
            pid_to_index=args.pid_to_index, direction=args.direction,
            max_angle=args.max_angle, unique_seeds=True, test_frenkel=True,
            zero_nonfrenkel=args.zero_nonfrenkel, verbosity=args.verbosity,
            adaptive_timestep=not args.constant_timestep,
            max_displacement=args.max_displacement, uid=args.jid)
