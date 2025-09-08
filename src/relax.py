import argparse
import json
import os

import defexp
import materials

if __name__ == "__main__":
    print("Running relax.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material label")
    parser.add_argument("-b", "--binary", type=str, default=None)
    parser.add_argument(
            "--pot", type=str, default="", help="potential file name")
    parser.add_argument("-c", "--configpath", type=str, default=defexp.default_config_path())
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

    proj = os.getenv("PROJ", default=".")
    resdir = "/".join((proj, "thresholds", args.material))

    material = materials.load_material(f"{args.configpath}/materials", args.material)
    sim_info = defexp.load_sim_info(f"{args.configpath}/simulations", args.material)
    energies = np.loadtxt(f"{args.configpath}/energies/{args.material}.dat")

    lattice = material.lattice(sim_info["repeat"])

    exp_io = defexp.ExperimentIO(
            material.label, res_dir=resdir, save_thermo=["Time", "PotEng"])
    lammps_io = defexp.LAMMPSIO(material.label)

    simulation = defexp.RelaxSimulation(
            lammps_binary, lattice, lammps_io, time_lammps=True,
            timestep=sim_info["timestep"], duration=sim_info["impact_duration"],
            temperature=sim_info["temperature"])

    simulation.run()
