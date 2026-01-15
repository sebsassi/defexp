import argparse
import os
import glob
import typing

import numpy as np

def average_over_sites(
    data: dict[typing.Any, dict[typing.Any, np.ndarray]]
) -> dict[typing.Any, np.ndarray]:
    return {key: np.concatenate(list(value.values()), axis=0) for key, value in data.items()}


def create_histograms(
    data: np.ndarray | dict[typing.Any, np.ndarray],
    x: typing.Optional[np.ndarray] = None,
    bins: typing.Optional[tuple[int,int]] = None,
    bin_width: float = 1.0,
    density: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray | dict[typing.Any, np.ndarray]]:
    if isinstance(data, dict):
        xmax = np.max([np.max(value[:,2]) for value in data.values()])
        ymax = np.max([np.max(value[:,3]) for value in data.values()])
    else:
        xmax = np.max(data[:,2])
        ymax = np.max(data[:,3])

    if bins is None:
        bins = (int(np.ceil(xmax/bin_width)), int(np.ceil(ymax/bin_width)))
    else:
        bins[0] = int(np.ceil(xmax/bin_width)) if bins[0] is None else bins[0]
        bins[1] = int(np.ceil(ymax/bin_width)) if bins[1] is None else bins[1]

    if x is None:
        x = np.linspace(0.0, xmax, bins[0])
    else:
        bins[0] = x

    bin_ranges = [[x[0], x[-1]], [0.0, ymax]]
    if isinstance(data, dict):
        histogram = {key: np.histogram2d(value[:,2],value[:,3], bins=bins, range=bin_ranges, density=density)[0] for key, value in data.items()}
    else:
        histogram = np.histogram2d(data[:,2],data[:,3])[0]

    y = np.linspace(0.0, ymax, bins[1])
    return x, y, histogram


def energy_loss_statistics(
    function: typing.Callable,
    data: np.ndarray | dict[typing.Any, np.ndarray],
    energies: typing.Optional[np.ndarray] = None,
    bin_width: float = 1.0
):
    if energies is None:
        if isinstance(data, dict):
            emax = np.max([np.max(value[:,2]) for value in data.values()])
        else:
            emax = np.max(data[:,2])
        energies = np.linspace(0.0, emax, int(np.ceil(emax/bin_width)))

    if isinstance(data, dict):
        return {key: average_energy_loss(value) for key, value in data.items()}
    else:
        indices = np.digitize(data[:,2], energies)
        return np.array([function(data[:,3][indices == i]) for i in range(energies.size)])


def average_energy_loss(
    data: np.ndarray | dict[typing.Any, np.ndarray],
    energies: typing.Optional[np.ndarray] = None,
    bin_width: float = 1.0
):
    return energy_loss_statistics(np.mean, data, energies, bin_width)


def minimum_energy_loss(
    data: np.ndarray | dict[typing.Any, np.ndarray],
    energies: typing.Optional[np.ndarray] = None,
    bin_width: float = 1.0
):
    return energy_loss_statistics(np.min, data, energies, bin_width)


def maximum_energy_loss(
    data: np.ndarray | dict[typing.Any, np.ndarray],
    energies: typing.Optional[np.ndarray] = None,
    bin_width: float = 1.0
):
    return energy_loss_statistics(np.max, data, energies, bin_width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material label")
    parser.add_argument("--extra-label", type=str, default=None, help="extra label on data")
    args = parser.parse_args()

    if args.extra_label is None:
        dirname = f"{os.getenv("HOME")}/mdsim/remote_data/eloss/{args.material}"
        filenames = glob.glob(f"{dirname}/eloss_{args.material}_*.dat")
    else:
        dirname = f"{os.getenv("HOME")}/mdsim/remote_data/eloss/{args.material}/{args.extra_label}"
        filenames = glob.glob(f"{dirname}/eloss_{args.material}_{args.extra_label}_*.dat")

    data = {}
    for filename in filenames:
        element, index = filename.removesuffix(".dat").rsplit("_", maxsplit=2)[1:]
        if element not in data.keys():
            data[element] = {}
        data[element][int(index)] = np.loadtxt(filename)

    site_averaged_data = average_over_sites(data)

    energies, eloss, histograms = create_histograms(site_averaged_data)

    min_eloss = minimum_energy_loss(site_averaged_data, energies)
    max_eloss = maximum_energy_loss(site_averaged_data, energies)
    avg_eloss = average_energy_loss(site_averaged_data, energies)

    processed_dir = f"{os.getenv("HOME")}/mdsim/processed"
    if not os.path.isdir(processed_dir): os.mkdir(processed_dir)

    material_dir = f"{processed_dir}/{args.material}"
    if not os.path.isdir(material_dir): os.mkdir(material_dir)

    if args.extra_label is None:
        out_dir = material_dir
    else:
        out_dir = f"{material_dir}/{args.extra_label}"
        if not os.path.isdir(label_dir): os.mkdir(label_dir)


    for element in data.keys():
        np.savetxt(f"{out_dir}/eloss_statistics_{element}.dat", np.column_stack((energies, min_eloss[element], max_eloss[element], avg_eloss[element])))
        np.savez(f"{out_dir}/eloss_histogram_{element}.npz", energies=energies, eloss=eloss, histogram=histograms[element])


if __name__ == "__main__":
    main()
