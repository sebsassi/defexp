import argparse

import numpy as np
import matplotlib.pyplot as plt


def spherical_to_cartesian(directions: np.ndarray):
    if (directions.ndim > 1):
        return np.stack(
            (np.sin(directions[...,0])*np.cos(directions[...,1]),
                np.sin(directions[...,0])*np.sin(directions[...,1]),
                np.cos(directions[...,0])),
            axis=-1)
    elif (directions.ndim == 1):
        return np.array([
            np.sin(directions[0])*np.cos(directions[1]),
            np.sin(directions[0])*np.sin(directions[1]),
            np.cos(directions[0])])
    else:
        raise RuntimeError("Invalid direction")


def cone_mask(
    directions: np.ndarray, central_direction: np.ndarray, aperture: float
) -> np.ndarray:
    return np.vecdot(spherical_to_cartesian(directions), spherical_to_cartesian(central_direction)) > np.cos(aperture)


def cone_mean_and_standard_dev(
    directions: np.ndarray, values: np.ndarray, central_directions: np.ndarray,
    aperture: float
) -> np.ndarray:
    central_directions_flat = np.reshape(central_directions, (-1,2))
    mean = np.zeros(central_directions_flat.shape[0])
    stdev = np.zeros(central_directions_flat.shape[0])
    count = np.zeros(central_directions_flat.shape[0])
    for i, central_direction in enumerate(central_directions_flat):
        cone_values = values[cone_mask(directions, central_direction, aperture)]
        mean[i] = np.mean(cone_values)
        stdev[i] = np.std(cone_values)
        count[i] = cone_values.size
    return np.reshape(mean, central_directions.shape[:-1]), np.reshape(stdev, central_directions.shape[:-1]), np.reshape(count, central_directions.shape[:-1])


def generate_cone_mean_stdev_grid(
    directions: np.ndarray, values: np.ndarray, aperture: float, xsize: int, ysize: int
):
    pa = np.linspace(0,np.pi,ysize)
    az = np.linspace(0,2.0*np.pi,xsize)

    azg, pag = np.meshgrid(az, pa)

    central_directions = np.stack((pag, azg), axis=-1)
    return az, pa, *cone_mean_and_standard_dev(directions, values, central_directions, aperture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="name of data file")
    parser.add_argument("aperture", type=float, help="angular size of cone in radians")
    parser.add_argument("gridsize", type=int, nargs=2, help="size of sampling grid")
    args = parser.parse_args()

    data = np.loadtxt(args.filename)
    directions = data[:,1:3]
    values = data[:,4]

    az, pa, mean, stdev, counts = generate_cone_mean_stdev_grid(
            directions, values, args.aperture, args.gridsize[0], args.gridsize[1])

    fig, ax = plt.subplots(3)

    for axis in ax:
        axis.set_xlabel(r"$\varphi$", size=16)
        axis.set_ylabel(r"$\theta$", size=16)

    ax[0].set_title("Energy loss mean")
    ax[1].set_title("Energy loss stdev")
    ax[2].set_title("Number of samples")

    mean_mesh = ax[0].pcolormesh(az, pa, mean)
    stdev_mesh = ax[1].pcolormesh(az, pa, stdev)
    counts_mesh = ax[2].pcolormesh(az, pa, counts)

    fig.colorbar(mean_mesh, ax=ax[0])
    fig.colorbar(stdev_mesh, ax=ax[1])
    fig.colorbar(counts_mesh, ax=ax[2])

    plt.show()

