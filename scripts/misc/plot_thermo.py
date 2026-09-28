import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt


def load_thermo(dir: str, material: str, label: str):
    if label is None:
        filenames = glob.glob(f"{dir}/{material}/*.npz")
    else:
        filenames = glob.glob(f"{dir}/{material}/{label}/*.npz")

    return {filename: np.load(filename) for filename in filenames}


def find_column(columns: np.ndarray, column):
    for i, candidate in enumerate(columns):
        if candidate == column:
            return i


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material name")
    parser.add_argument("-l", "--label", type=str, default=None, help="simulation label")
    parser.add_argument("-t", "--thermo-dir", type=str, default="./thermo", help="thermo data directory")
    parser.add_argument("-c", "--columns", type=str, nargs="+", help="columns to plot")
    args = parser.parse_args()

    thermo = load_thermo(args.thermo_dir, args.material, args.label)
    print(thermo)

    fig = plt.figure()
    axs = fig.subplots(len(args.columns))

    if len(args.columns) == 1:
        axs = [axs]

    for ax, column in zip(axs, args.columns):
        ax.set_xlabel("Time")
        ax.set_ylabel(column)

    for filename, data in thermo.items():
        time_index = find_column(data["columns"], "Time")
        for i, column in enumerate(args.columns):
            column_index = find_column(data["columns"], column)
            axs[i].plot(data["thermo"][:,time_index], data["thermo"][:,column_index], label=filename)

    plt.show()
