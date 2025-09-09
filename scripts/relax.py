import argparse
import json
import os

import defexp

if __name__ == "__main__":
    print("Running relax.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material label")
    parser.add_argument("-b", "--binary", type=str, default=None)
    parser.add_argument(
            "--pot", type=str, default="", help="potential file name")
    parser.add_argument("-c", "--configpath", type=str, default=defexp.default_config_path())
    parser.add_argument("--work-dir", type=str, default=".", help="output directory for intermediate/auxillary files")
    args = parser.parse_args()

    if args.binary is not None:
        lammps_binary = args.binary
    else:
        lammps_binary = os.getenv("LMP_BINARY", default=None)
    if lammps_binary is None:
        raise RuntimeError(
                "Missing LAMMPS binary. Either provide name of the LAMMPS "
                "binary in the relevant command line argument or set the "
                "environment variable LMP_BINARY.")

    res_dir = f"{args.res_dir}/eloss/{args.material}"
    lmp_dir = f"{args.work_dir}/lammps_work"
    dump_dir = f"{args.work_dir}/dump"
    thermo_dir = f"{args.work_dir}/thermo"
    log_dir = f"{args.work_dir}/logs"

    material = materials.load_material(f"{args.configpath}/materials", args.material)
    sim_info = defexp.load_sim_info(f"{args.configpath}/simulations", args.material)
    energies = np.loadtxt(f"{args.configpath}/energies/{args.material}.dat")

    lattice = material.lattice(sim_info["repeat"])

    label = f"relax_{material.label}"
    exp_io = defexp.ExperimentIO(
            label, res_dir, thermo_dir, log_dir, save_thermo=["Time", "PotEng"])
    lammps_io = defexp.LAMMPSIO(label, lmp_dir, dump_dir)

    exp_io.make_dirs()
    lammps_io.make_dirs()

    simulation = defexp.RelaxSimulation(
            lammps_binary, lattice, lammps_io, time_lammps=True,
            timestep=sim_info["timestep"], duration=sim_info["impact_duration"],
            temperature=sim_info["temperature"])

    simulation.run()
