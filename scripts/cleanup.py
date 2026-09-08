import os
import os.path
import glob

if __name__ == "__main__":
    print("Running cleanup.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input-file", type=str, default=None, help="JSON file providing same parameters as the command line (command line arguments override values in the file)")
    parser.add_argument("-l", "--label", type=str, default=None, help="experiment label")
    parser.add_argument("-m", "--material", type=str, default=None, "material name")
    parser.add_argument(      "--remove-res", action="store_true", help="remove result files")
    parser.add_argument("-R", "--res-dir", type=str, default=".", help="output directory for main results")
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

    base_dirs = {
        lmp: f"{args.work_dir}/lammps_work"
        dump: f"{args.work_dir}/dump"
        res: f"{args.res_dir}/eloss"
        thermo: f"{args.work_dir}/thermo"
        log: f"{args.work_dir}/logs"
    }
    material_dirs = {k: f"{dir}/{args.material}" for k, dir in base_dirs.items()}

    if args.label is None:
        input_dirs = material_dirs
    else:
        input_dirs = {k: f"{dir}/{args.label}" for k, dir in material_dirs.items()}

    for filename in glob.glob(f"{input_dirs["lmp"]}/*.log"): os.remove(filename)
    for filename in glob.glob(f"{input_dirs["dump"]}/*.dump"): os.remove(filename)
    for filename in glob.glob(f"{input_dirs["dump"]}/*.dump.gz"): os.remove(filename)
    for filename in glob.glob(f"{input_dirs["thermo"]}/*.npz"): os.remove(filename)
    for filename in glob.glob(f"{input_dirs["log"]}/*.log"): os.remove(filename)
    if args.remove_res:
        for filename in glob.glob(f"{input_dirs["res"]}/*.dat"): os.remove(filename)
