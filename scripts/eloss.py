import datetime
import argparse
import json
import os
import logging
import typing

import numpy as np

import defexp


def random_energy_loss(
    recoil_simulation: defexp.RecoilSimulation, seed: int, count: int,
    emin: float, emax: float, aind: int, pid: int,
    screen: typing.Optional[str] = None, verbosity: int = 1, **kwargs
):
    rng = np.random.default_rng(seed)
    alt, az = defexp.uniform_angles(rng, count)
    energies = emin + (emax - emin)*rng.random(count)
    unitv = defexp.angles_to_vec(alt, az)

    atom_type = recoil_simulation.lattice.atoms.numbers[aind]

    if "symbol" in recoil_simulation.lattice.material.atom_props[atom_type]:
        aid = recoil_simulation.lattice.material.atom_props[atom_type]["symbol"]
    else:
        aid = str(atom_type)

    result_fname = (f"{recoil_simulation.io.res_dir}/"
            f"random_energy_loss_{aid}_{aind}_{pid}.dat")
    defexp.ensure_file_ends_with_new_line(result_fname, verbosity)

    df_name, tf_name = recoil_simulation.lammps_io.create_data_and_thermo_file(pid)

    lammps_args = {"log": tf_name}
    if screen is not None:
        lammps_args["screen"] = screen

    for i in range(count):
        if verbosity in (1,2):
            defexp.log_print(f"Working on sample {i + 1:d}/{count:d}.")
        elif verbosity > 2:
            defexp.log_print(
                    f"Working on sample {i + 1:d}/{count:d}: "
                    f"{alt[i]:.5f} {az[i]:.5f} {energies[i]:.5e}.")

        depot, frenkel_defect = recoil_simulation.run(
                atom_type, aind, unitv[i], energies[i], df_name, lammps_args,
                tf_name, pid, **kwargs)

        with open(result_fname, "a") as f:
            if verbosity > 1:
                logging.debug(f"Opened file {result_fname} in append mode.")
            f.write(
                    f"{alt[i]:.16e} {az[i]:.16e} {energies[i]:.16e} "
                    f"{depot:.16e} {int(frenkel_defect)}\n")

        if verbosity > 1:
            logging.debug(f"Wrote to file {result_fname}.")


def execute(
    recoil_simulation: defexp.RecoilSimulation, seed: int, count: int,
    emin: float, emax: float, aind: int, pid: int, parallel: bool = False,
    unique_seeds: bool = True, **kwargs
):
    if parallel:
        processes = []
        ainds = simulation.lattice.indices_in_central_cell(lammps=False)
        for i, aind, atom_type in enumerate(zip(ainds, atom_types)):
            process_seed = abs(hash((seed, i))) if unique_seeds else seed
            args = (recoil_simulation, process_seed, count, emin, emax, aind, i)
            processes.append(mp.Process(target=random_angle_thresholds, args=args, kwargs=kwargs))
        for p in processes: p.start()
        for p in processes: p.join()
    else:
        process_seed = abs(hash((seed, pid))) if unique_seeds else seed
        random_energy_loss(
                recoil_simulation, process_seed, count, emin, emax, aind, pid, **kwargs)


if __name__ == "__main__":
    print("Running thresholds.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material label")
    parser.add_argument("jid", type=int, help="job ID")
    parser.add_argument("i", type=int, help="index of recoil atom")
    parser.add_argument("seed", type=int, help="input rng seed")
    parser.add_argument("count", type=int, help="number of recoil experiments")
    parser.add_argument("-b", "--binary", type=str, default=None)
    parser.add_argument("-c", "--configpath", type=str, default=".")
    parser.add_argument("-d", "--res-dir", type=str, default=".", help="output directory for main results")
    parser.add_argument("--work-dir", type=str, default=".", help="output directory for intermediate/auxillary files")
    parser.add_argument("-z", "--zero-nonfrenkel", action="store_true", help="set energy loss to zero if there are no Frenkel defects")
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
    logging.info(f"Result directory: {args.res_dir}")
    logging.info(f"Work directory: {args.work_dir}")
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
            lammps_binary, lattice, exp_io, lammps_io, dump=False,
            time_lammps=True, timestep=sim_info["timestep"], 
            duration=sim_info["impact_duration"],
            temperature=sim_info["temperature"])

    ainds = lattice.indices_in_central_cell(lammps=False)
    atom_index = ainds[args.i % material.unit_cell_atoms.shape[0]]
    logging.info(f"Atom index: {atom_index}")

    execute(
            simulation, args.seed, args.count, sim_info["emin"], sim_info["emax"],
            atom_index, args.i, unique_seeds=True, test_frenkel=True, smooth_count=10,
            zero_nonfrenkel=args.zero_nonfrenkel)
