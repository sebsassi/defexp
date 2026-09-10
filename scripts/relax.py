import argparse
import json
import os

import numpy as np

import defexp

if __name__ == "__main__":
    print("Running relax.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config-dir", type=str, default=".", help="directory containing material/simulation configuration files")
    parser.add_argument("-I", "--input-file", type=str, default=None, help="JSON file providing same parameters as the command line (command line arguments override values in the file)")
    parser.add_argument("-l", "--label", type=str, default=None, help="experiment label")
    parser.add_argument("-m", "--material", type=str, default=None, help="material name")
    parser.add_argument(      "--relax-duration", type=float, default=2.0, help="simulation duration in picoseconds")
    parser.add_argument(      "--repeat", type=float, nargs=3, default=None, help="number of repeated unit cells along each axis")
    parser.add_argument("-R", "--res-dir", type=str, default=".", help="output directory for main results")
    parser.add_argument(      "--temperature", type=float, default=None, help="temperature of the system")
    parser.add_argument("-T", "--timestep", type=float, default=None, help="simulation timestep in picoseconds")
    parser.add_argument("-W", "--work-dir", type=str, default=".", help="output directory for intermediate/auxillary files")
    args = parser.parse_args()

    if args.input_file is not None:
        with open(args.input_file, "r") as f:
            arguments = json.load(f)

        for key, value in arguments.items():
            if key in vars(args).keys():
                if getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)

    if args.material is None:
        raise RuntimeError("Argument `material` needs to be defined either in an input file or via the command line.")
    if args.repeat is None:
        raise RuntimeError("Argument `repeat` needs to be defined either in an input file or via the command line.")
    if args.temperature is None:
        raise RuntimeError("Argument `temperature` needs to be defined either in an input file or via the command line.")
    if args.timestep is None:
        raise RuntimeError("Argument `timestep` needs to be defined either in an input file or via the command line.")

    base_dirs = {
        lmp: f"{args.work_dir}/lammps_work"
        dump: f"{args.work_dir}/dump"
        res: f"{args.res_dir}/eloss"
        thermo: f"{args.work_dir}/thermo"
        log: f"{args.work_dir}/logs"
    }
    for dir in base_dirs.values():
        if not os.path.isdir(dir): os.mkdir(dir)

    material_dirs = {k: f"{dir}/{args.material}" for k, dir in base_dirs.items()}
    for dir in material_dirs.values():
        if not os.path.isdir(f"{dir}"): os.mkdir(f"{dir}")

    if args.label is None:
        input_dirs = material_dirs
    else:
        input_dirs = {k: f"{dir}/{args.label}" for k, dir in material_dirs.items()}
        for dir in input_dirs.values():
            if not os.path.isdir(f"{dir}"): os.mkdir(f"{dir}/")


    material = defexp.load_material(f"{args.config_dir}/materials", f"{args.config_dir}/potentials", args.material)

    lattice = defexp.Lattice(material, args.repeat)

    label = "relax"
    exp_io = defexp.ExperimentIO(label, input_dirs["res"], input_dirs["thermo"], input_dirs["log"])
    lammps_io = defexp.LAMMPSIO(label, input_dirs["lmp"], input_dirs["dump"])

    simulation = defexp.RelaxSimulation(
            lattice, lammps_io, time_lammps=True, timestep=args.timestep,
                duration=args.relax_duration, temperature=args.temperature)

    simulation.run(verbosity=2)
