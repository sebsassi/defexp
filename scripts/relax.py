import argparse
import json
import os

import numpy as np

import defexp

if __name__ == "__main__":
    print("Running relax.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material name")
    parser.add_argument("-b", "--lmp-binary", type=str, default=None, help="name of the LAMMPS binary")
    parser.add_argument("-d", "--res-dir", type=str, default=".", help="output directory for main results")
    parser.add_argument("-c", "--config-dir", type=str, default=".", help="directory containing material/simulation configuration files")
    parser.add_argument("--work-dir", type=str, default=".", help="output directory for intermediate/auxillary files")
    args = parser.parse_args()

    if args.lmp_binary is not None:
        lmp_binary = args.lmp_binary
    else:
        lmp_binary = os.getenv("LMP_BINARY", default=None)
    if lmp_binary is None:
        raise RuntimeError(
                "Missing LAMMPS binary. Either provide name of the LAMMPS "
                "binary in the relevant command line argument or set the "
                "environment variable LMP_BINARY.")

    lmp_dir = f"{args.work_dir}/lammps_work"
    dump_dir = f"{args.work_dir}/dump"
    res_dir = f"{args.res_dir}/eloss/{args.material}"
    thermo_dir = f"{args.work_dir}/thermo"
    log_dir = f"{args.work_dir}/logs"

    if not os.path.isdir(lmp_dir): os.mkdir(lmp_dir)
    if not os.path.isdir(dump_dir): os.mkdir(dump_dir)
    if not os.path.isdir(os.path.dirname(res_dir)): os.mkdir(os.path.dirname(res_dir))
    if not os.path.isdir(res_dir): os.mkdir(res_dir)
    if not os.path.isdir(thermo_dir): os.mkdir(thermo_dir)
    if not os.path.isdir(log_dir): os.mkdir(log_dir)

    material = defexp.load_material(f"{args.config_dir}/materials", f"{args.config_dir}/potentials", args.material)
    sim_info = defexp.load_sim_info(f"{args.config_dir}/sim_info", args.material)
    energies = np.loadtxt(f"{args.config_dir}/energies/{args.material}.dat")

    lattice = defexp.Lattice(material, sim_info["repeat"])

    label = f"relax_{material.label}"
    lammps_io = defexp.LAMMPSIO(label, lmp_dir, dump_dir)

    exp_io.make_dirs()
    lammps_io.make_dirs()

    simulation = defexp.RelaxSimulation(
            lmp_binary, lattice, lammps_io, time_lammps=True,
            timestep=sim_info["timestep"], duration=sim_info["impact_duration"],
            temperature=sim_info["temperature"])

    simulation.run(verbosity=2)
