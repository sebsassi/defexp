import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from cone_sampler import generate_cone_mean_stdev_grid

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


def cartesian_to_spherical(directions: np.ndarray):
    length = np.linalg.vector_norm(directions, axis=-1)
    if (directions.ndim > 1):
        pa = np.arccos(directions[:,2]/length)
        az = np.atan2(directions[:,1], directions[:,0])
        return np.stack((pa, az), axis=-1)
    elif (directions.ndim == 1):
        pa = np.arccos(directions[2]/length)
        az = np.atan2(directions[1], directions[0])
        return bp.array([pa, az])
    else:
        raise RuntimeError("Invalid direction")


def uniform_angles(
    rng: np.random.Generator, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate angles from a spherically uniform distribution.

    Parameters
    ----------
    seed : int
        Seed for the RNG.
    count : int
        Number of angles to generate

    Returns
    -------
    numpy.ndarray
        Polar angles.
    numpy.ndarray
        Azimuth angles.
    """
    pa = np.arccos(2*rng.random(count) - 1)
    az = (2*np.pi)*rng.random(count)
    return pa, az


def truncated_gaussian(mu: float, sigma: float, count: int) -> np.ndarray:
    a = -mu/sigma
    b = np.inf
    return stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=count)


def antiprism_dupe(directions: np.ndarray):
    generators = {
        "reflection": np.array([
            [-1, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 1],
        ]),
        "rotation": np.array([
            [np.cos(2*np.pi/3), -np.sin(2*np.pi/3), 0],
            [np.sin(2*np.pi/3),  np.cos(2*np.pi/3), 0],
            [                0,                  0, 1]
        ]),
        "rotoreflection": np.array([
            [np.cos(2*np.pi/6), -np.sin(2*np.pi/6),  0],
            [np.sin(2*np.pi/6),  np.cos(2*np.pi/6),  0],
            [                0,                  0, -1]
        ])
    }

    cartesian_I = spherical_to_cartesian(directions)
    cartesian_R = np.matvec(generators["rotation"], cartesian_I)
    cartesian_R2 = np.matvec(generators["rotation"], cartesian_R)
    cartesian_P = np.matvec(generators["reflection"], cartesian_I)
    cartesian_PR = np.matvec(generators["reflection"], cartesian_R)
    cartesian_PR2 = np.matvec(generators["reflection"], cartesian_R2)
    cartesian_Q = np.matvec(generators["rotoreflection"], cartesian_I)
    cartesian_QR = np.matvec(generators["rotoreflection"], cartesian_R)
    cartesian_QR2 = np.matvec(generators["rotoreflection"], cartesian_R2)
    cartesian_QP = np.matvec(generators["rotoreflection"], cartesian_P)
    cartesian_QPR = np.matvec(generators["rotoreflection"], cartesian_PR)
    cartesian_QPR2 = np.matvec(generators["rotoreflection"], cartesian_PR2)
    return [
        cartesian_to_spherical(cartesian_I),
        cartesian_to_spherical(cartesian_R),
        cartesian_to_spherical(cartesian_R2),
        cartesian_to_spherical(cartesian_P),
        cartesian_to_spherical(cartesian_PR),
        cartesian_to_spherical(cartesian_PR2),
        cartesian_to_spherical(cartesian_Q),
        cartesian_to_spherical(cartesian_QR),
        cartesian_to_spherical(cartesian_QR2),
        cartesian_to_spherical(cartesian_QP),
        cartesian_to_spherical(cartesian_QPR),
        cartesian_to_spherical(cartesian_QPR2),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int)
    args = parser.parse_args()

    mu = 30.0
    sigma = 6.0

    directions = np.column_stack(uniform_angles(np.random.default_rng(), args.count))
    values = truncated_gaussian(mu, sigma, args.count)

    duped_directions = np.concatenate(antiprism_dupe(directions), axis=0)
    duped_values = np.concatenate(12*[values], axis=0)

    aperture = 0.2
    xsize = 160
    ysize = 80

    az, pa, mean, stdev, counts = generate_cone_mean_stdev_grid(
            directions, values, aperture, xsize, ysize)

    az, pa, duped_mean, duped_stdev, duped_counts = generate_cone_mean_stdev_grid(
            duped_directions, duped_values, aperture, xsize, ysize)

    fig, ax = plt.subplots(2, 2)

    mean_mesh = ax[0][0].pcolormesh(az, pa, mean)
    duped_mean_mesh = ax[0][1].pcolormesh(az, pa, duped_mean)
    stdev_mesh = ax[1][0].pcolormesh(az, pa, stdev)
    duped_stdev_mesh = ax[1][1].pcolormesh(az, pa, duped_stdev)

    fig.colorbar(mean_mesh, ax=ax[0][0])
    fig.colorbar(duped_mean_mesh, ax=ax[0][1])
    fig.colorbar(stdev_mesh, ax=ax[1][0])
    fig.colorbar(duped_stdev_mesh, ax=ax[1][1])

    plt.show()
