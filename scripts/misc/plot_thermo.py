import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

def load_thermo(dir: str, material: str, label: str) -> dict[str, np.ndarray]:
    if label is None:
        filenames = glob.glob(f"{dir}/{material}/*.npz")
    else:
        filenames = glob.glob(f"{dir}/{material}/{label}/*.npz")

    return {filename: np.load(filename) for filename in filenames}


def find_column(columns: np.ndarray, column) -> int:
    for i, candidate in enumerate(columns):
        if candidate == column:
            return i


def extract_hash(filename: str) -> int:
    return int(filename.removesuffix(".dat.npz").rsplit("_")[-1])


def optimal_layout(size: int) -> tuple[int, int]:
    x = np.ceil(np.sqrt(size))
    y = np.ceil(size/x)
    return int(x), int(y)


def per_displacement(dir: str, material: str, label_base: str, displacements: list[float], column: str):
    thermo = {d: load_thermo(dir, material, f"{label_base}_{d:.4f}a") for d in displacements}
    thermo = {d: {extract_hash(filename): data for filename, data in t.items()} for d, t in thermo.items()}
    hashes = list(set(hash for t in thermo.values() for hash in t.keys()))

    fig = plt.figure()
    axs = fig.subplots(*optimal_layout(len(hashes)))
    axes_map = {hash: ax for hash, ax in zip(hashes, axs.flatten())}

    for ax in axs.flatten():
        ax.set_xlabel("Time")
        ax.set_xlabel(column)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_map = {d: color for d, color in zip(displacements, colors)}

    custom_lines = {d: lines.Line2D([0], [0], color=c) for d, c in colors_map.items()}
    plt.figlegend(
        custom_lines.values(),
        custom_lines.keys(),
        loc="upper center", ncols=6)

    for d, t in thermo.items():
        for hash, data in t.items():
            time_index = find_column(data["columns"], "Time")
            column_index = find_column(data["columns"], column)
            axes_map[hash].plot(data["thermo"][:,time_index], data["thermo"][:,column_index], label=d, color=colors_map[d])

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material name")
    parser.add_argument("-l", "--label", type=str, default=None, help="simulation label")
    parser.add_argument("-t", "--thermo-dir", type=str, default="./thermo", help="thermo data directory")
    parser.add_argument("-c", "--columns", type=str, nargs="+", help="columns to plot")
    parser.add_argument(      "--per-displacement", action="store_true", help="plot per displacement")
    parser.add_argument("-d", "--displacements", type=float, nargs="+", help="plot per displacement")
    args = parser.parse_args()

    if args.per_displacement:
        per_displacement(args.thermo_dir, args.material, args.label, args.displacements, args.columns[0])
    else:
        raise RuntimeError("Use --per-displacement flag")
