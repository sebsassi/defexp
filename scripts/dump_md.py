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

    config_fname = f"{args.configpath}/{args.material}_config.json"
    with open(config_fname,"r") as f:
        config = json.load(f)
        if config["material"] != args.material:
            raise ValueError(
                    "Material in config file differs from argument material. "
                    f"Is {config_fname} the desired config file?")

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
    dumpdir = os.getenv("DUMP_DIR", default="dump")

    if "pot_file" in config:
        material = construct(args.material, pot_file=config["pot_file"])
    else:
        material = construct(args.material)

    energies = np.array([args.energy])*1.0e-9

    ainds = material.indices_in_central_cell(lammps=False)
    
    io = defexp.ExperimentIO(material.label, res_dir=resdir, dump_dir=dumpdir)

    print(ainds)
    if args.relax:
        defexp.RelaxExperiment(lammps_binary, material, io).run()

    experiment = defexp.DefectExperiment(
            lammps_binary, material, io, dump=True,
            verbosity=2, screen="", 
            time_lammps=True, timestep=config["timestep"], 
            duration=config["impact_duration"],
            temperature=config["temperature"])

    extra_label = f"{1000*config['timestep']:.2f}_{config['impact_duration']:.1f}_dump_{args.energy:.0f}"
    atom_type = material.lattice.numbers[args.aind]
    experiment.scan_energy_range(
            atom_type, args.aind, defexp.angles_to_vec(args.alt, args.az),
            energies, extra_label, test_frenkel=False)
