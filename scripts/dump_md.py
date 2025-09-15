from materials import *

import argparse
import json
import logging

if __name__ == "__main__":
    print("Running dump_md.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material label")
    parser.add_argument("aind", type=int, help="index of recoil atom")
    parser.add_argument("alt", type=float, help="altitude of recoil direction")
    parser.add_argument("az", type=float, help="azimuth of recoil direction")
    parser.add_argument("energy", type=float, help="recoil energy")
    parser.add_argument("-b", "--binary", type=str, default=None)
    parser.add_argument(
            "--norelax", dest="relax", action="store_false", help="skip relax")
    parser.add_argument("-c", "--configpath", type=str, default=defexp.default_config_path())
    args = parser.parse_args()

    logging.basicConfig(filename=f"dump.log", level=logging.DEBUG)
    logging.info(f"Material: {args.material}")
    logging.info(f"Atom index: {args.aind}")
    logging.info(f"Azimuth: {args.az:.5f}")
    logging.info(f"Altitude: {args.alt:.5f}")
    logging.info(f"Energy: {args.energy:.5e}")
    logging.info(f"Config path: {args.configpath}")

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

    material = defexp.load_material(f"{args.configpath}/materials", f"{args.configpath}/potentials", args.material)
    sim_info = defexp.load_sim_info(f"{args.configpath}/sim_info", args.material)

    lattice = defexp.Lattice(material, sim_info["repeat"])

    label = f"eloss_{material.label}"
    exp_io = defexp.ExperimentIO(
            label, res_dir, thermo_dir, log_dir, save_thermo=["Time", "PotEng"])
    lammps_io = defexp.LAMMPSIO(label, lmp_dir, dump_dir)

    simulation = defexp.RecoilSimulation(
            lammps_binary, lattice, exp_io, lammps_io, dump=True,
            time_lammps=True, timestep=sim_info["timestep"], 
            duration=sim_info["impact_duration"],
            temperature=sim_info["temperature"])

    ainds = lattice.indices_in_central_cell(lammps=False)
    atom_index = ainds[args.i % material.unit_cell_atoms.shape[0]]
    logging.info(f"Atom index: {atom_index}")

    print(ainds)
    if args.relax:
        defexp.RelaxSimulation(
            lammps_binary, lattice, lammps_io, time_lammps=True,
            timestep=sim_info["timestep"], duration=sim_info["impact_duration"],
            temperature=sim_info["temperature"]).run(verbosity=2)

    extra_label = f"{1000*sim_info['timestep']:.2f}_{sim_info['impact_duration']:.1f}_dump_{args.energy:.0f}"
    atom_type = material.lattice.numbers[args.aind]
