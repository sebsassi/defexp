import argparse
import os

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_eloss(fname: str):
    return np.loadtxt(fname)


def healpix_stats(
    colat: np.ndarray, lon: np.ndarray, eloss: np.ndarray, nside: int
) -> tuple[np.ndarray, np.ndarray]:
    pixels = hp.pixelfunc.ang2pix(nside, colat, lon)
    means = np.zeros(12*nside**2)
    stdevs = np.zeros(12*nside**2)
    for i in range(means.size):
        means[i] = np.mean(eloss[pixels == i])
        stdevs[i] = np.std(eloss[pixels == i])

    return means, stdevs


def plot(energies: list[int], data: dict, colat: np.ndarray, lon: np.ndarray):
    fig_map = plt.figure()
    top_grid = gridspec.GridSpec(1, 2, figure=fig_map)
    mean_map_grid = top_grid[0].subgridspec(5, 2, hspace=0, wspace=0)
    stdev_map_grid = top_grid[1].subgridspec(5, 2, hspace=0, wspace=0)

    mean_axes = mean_map_grid.subplots(sharex="col", sharey="row")
    stdev_axes = stdev_map_grid.subplots(sharex="col", sharey="row")

    print(f"{"l":10s}", "".join([f"{l:10d}" for l in range(data["power"]["mean"][0].size)]))
    for i, energy in enumerate(energies):
        print(f"{f"{energy} eV":10s}", "".join([f"{power:10.5f}" for power in data["power"]["mean"][i]]))

    for i, energy in enumerate(energies):
        mean_axes[i//2][i % 2].pcolormesh(lon, colat, data["map"]["mean"][i], vmin=0.0)
        mean_axes[i//2][i % 2].text(0, 0, f"{energy} eV", ha="left", va="bottom", color="white", alpha=0.7, fontfamily="monospace", fontsize=20)
        stdev_axes[i//2][i % 2].pcolormesh(lon, colat, data["map"]["stdev"][i], vmin=0.0)
        stdev_axes[i//2][i % 2].text(0, 0, f"{energy} eV", ha="left", va="bottom", color="white", alpha=0.7, fontfamily="monospace", fontsize=20)

    fig_map.tight_layout()

    fig_power, ax = plt.subplots(1, 2)
    ax[0].imshow(np.array(data["power"]["mean"]), norm="log")
    ax[1].imshow(np.array(data["power"]["stdev"]), norm="log")
    fig_power.tight_layout()
    plt.show()


def eloss_analysis(eloss_dir: str, material: str, element: str, energies: list[int]):
    anal_data = {
        "map": {
            "mean": [],
            "stdev": []
        },
        "power": {
            "mean": [],
            "stdev": []
        },
    }

    map_colat = np.linspace(0, np.pi, 100)
    map_lon = np.linspace(0, 2*np.pi, 200)

    map_colat_g, map_lon_g = np.meshgrid(map_colat, map_lon)

    nside = 4
    for energy in energies:
        fname = f"{eloss_dir}/{material}/{energy}eV_{element}_recoil/eloss_{material}_{energy}eV_{element}_recoil_{element}.dat"
        data = np.loadtxt(fname)
        colat = data[:,1]
        lon = data[:,2]
        eloss = data[:,4]
        means, stdevs = healpix_stats(colat, lon, eloss, nside)

        mean_power, mean_coeffs = hp.sphtfunc.anafast(means, lmax=2*nside, use_weights=True, alm=True)
        stdev_power, stdev_coeffs = hp.sphtfunc.anafast(stdevs, lmax=2*nside, use_weights=True, alm=True)

        anal_data["power"]["mean"].append(mean_power/mean_power[0])
        anal_data["power"]["stdev"].append(stdev_power/stdev_power[0])

        mean_upsampled = hp.sphtfunc.alm2map(mean_coeffs, 2*nside)
        stdev_upsampled = hp.sphtfunc.alm2map(stdev_coeffs, 2*nside)
        anal_data["map"]["mean"].append(hp.pixelfunc.get_interp_val(mean_upsampled, map_colat_g, map_lon_g))
        anal_data["map"]["stdev"].append(hp.pixelfunc.get_interp_val(stdev_upsampled, map_colat_g, map_lon_g))

    plot(energies, anal_data, map_colat_g, map_lon_g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material name")
    parser.add_argument("element", type=str, help="element symbol")
    parser.add_argument("energies", type=int, nargs="+", help="list of energy values")
    parser.add_argument("-d", "--data-dir", type=str, default=f"{os.getenv("PROJ")}/mdsim/remote_data/eloss", help="energy loss data directory")
    args = parser.parse_args()

    eloss_analysis(args.data_dir, args.material, args.element, args.energies)


