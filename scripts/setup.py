import argparse
import os

if __name__ == "__main__":
    print("Running setup.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--materials", type=str, nargs="+", default=None, help="material names")
    parser.add_argument("-W", "--work-dir", type=str, default=".", help="output directory for intermediate/auxillary files")
    parser.add_argument("-R", "--res-dir", type=str, default=".", help="output directory for main results")
    args = parser.parse_args()

    base_dirs = {
        "lmp": f"{args.work_dir}/lammps_work",
        "dump": f"{args.work_dir}/dump",
        "res": f"{args.res_dir}/eloss",
        "thermo": f"{args.work_dir}/thermo",
        "log": f"{args.work_dir}/logs"
    }
    for dir in base_dirs.values():
        if not os.path.isdir(dir): os.mkdir(dir)

    for material in args.materials:
        material_dirs = {k: f"{dir}/{material}" for k, dir in base_dirs.items()}
        for dir in material_dirs.values():
            if not os.path.isdir(f"{dir}"): os.mkdir(f"{dir}")
