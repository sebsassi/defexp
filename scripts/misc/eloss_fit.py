import argparse
import os
import glob

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def fit_func(eloss, p0, r0, p1, e1, s1, p2, e2, s2, p3, e3, s3):
    return (p0*np.exp(-r0*eloss)
            + p1*np.exp(-(eloss - e1)**2/(2*s1**2))/s1
            + p2*np.exp(-(eloss - e2)**2/(2*s2**2))/s2
            + p3*np.exp(-(eloss - e3)**2/(2*s3**2))/s3)/np.sqrt(2*np.pi)


def fit_func_der(eloss, p0, r0, p1, e1, s1, p2, e2, s2, p3, e3, s3):
    return np.array([
        -p0*eloss*np.exp(-r0*eloss),
        np.exp(-(eloss - e1)**2/(2*s1**2))/s1,
        np.exp(-(eloss - e1)**2/(2*s1**2))*(eloss - e1)/s1**2,
        np.exp(-(eloss - e1)**2/(2*s1**2))*((eloss - e1)**2 - s1**2)/s1**4,
        np.exp(-(eloss - e2)**2/(2*s2**2))/s2,
        np.exp(-(eloss - e2)**2/(2*s2**2))*(eloss - e2)/s2**2,
        np.exp(-(eloss - e2)**2/(2*s2**2))*((eloss - e2)**2 - s2**2)/s2**4,
        np.exp(-(eloss - e3)**2/(2*s3**2))/s3,
        np.exp(-(eloss - e3)**2/(2*s3**2))*(eloss - e3)/s3**2,
        np.exp(-(eloss - e3)**2/(2*s3**2))*((eloss - e3)**2 - s3**2)/s3**4
    ])


def opt_func(params, eloss, histogram):
    p0 = 1.0 - params[1] - params[4] - params[7]
    fit_value = fit_func(
            eloss,
            p0,
            params[0],
            params[1], params[2], params[3],
            params[4], params[5], params[6],
            params[7], params[8], params[9])
    f = np.sum((fit_value - histogram)**2)
    d_fit = fit_func_der(
            eloss,
            p0,
            params[0],
            params[1], params[2], params[3],
            params[4], params[5], params[6],
            params[7], params[8], params[9])
    df = 2.0*np.sum((fit_value - histogram)*d_fit, axis=-1)

    return f, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material", type=str, help="material label")
    parser.add_argument("--extra-label", type=str, default=None, help="extra label on data")
    args = parser.parse_args()

    if args.extra_label is None:
        filenames = glob.glob(f"{os.getenv("HOME")}/mdsim/processed/{args.material}/eloss_histogram_*.npz")
    else:
        filenames = glob.glob(f"{os.getenv("HOME")}/mdsim/processed/{args.material}/{args.extra_label}/eloss_histogram_*.npz")

    for filename in filenames:
        data = np.load(filename)
        histogram = data["histogram"]
        energies = data["energies"]
        eloss = data["eloss"]
        emax = np.max(eloss)

        bounds = ([0,       0,  1,      0.1,    0,  1,      0.1,    0,  1,      0.1 ],
                  [emax,    1,  emax,   emax,   1,  emax,   emax,   1,  emax,   emax])

        ineq_const = {
            "type": "ineq",
            "fun": lambda x: np.array([
                1.0 - x[1] - x[4] - x[7],
                x[5] - x[2],
                x[8] - x[5]]),
            "jac": lambda x: np.array([
                [0.0, -1.0,  0.0, 0.0, -1.0,  0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0,  0.0, -1.0, 0.0,  0.0,  1.0, 0.0,  0.0, 0.0, 0.0],
                [0.0,  0.0,  0.0, 0.0,  0.0, -1.0, 0.0,  0.0, 1.0, 0.0]])
        }

        params = []
        for i in range(histogram.shape[0]):
            res = opt.shgo(opt_func, args=(eloss, histogram[i]), bounds=bounds, workers=-1)
            params.append(res.x)
            init = res.x

        params = np.array(params)
        histogram_fit = np.array([[fit_func(eloss[i], *params[j]) for i in range(histogram.shape[1])] for j in range(histogram.shape[0])])

        fig, ax = plt.subplots(3)
        ax[0].imshow(histogram.T, aspect="auto")
        ax[1].imshow(histogram_fit.T, aspect="auto")

        line0, = ax[2].plot(energies, params[:,0], label="$p_0$", linestyle="-")
        ax[2].plot(energies, params[:,1], label="$\\tau$", color=line0.get_color(), linestyle="--")
        line1, = ax[2].plot(energies, params[:,2], label="$p_1$", linestyle="-")
        ax[2].plot(energies, params[:,3], label="$E_1$", color=line1.get_color(), linestyle="--")
        ax[2].plot(energies, params[:,4], label="$\\sigma_1$", color=line1.get_color(), linestyle=":")
        line2, = ax[2].plot(energies, params[:,5], label="$p_2$", linestyle="-")
        ax[2].plot(energies, params[:,6], label="$E_2$", color=line2.get_color(), linestyle="--")
        ax[2].plot(energies, params[:,7], label="$\\sigma_2$", color=line2.get_color(), linestyle=":")
        line3, = ax[2].plot(energies, params[:,8], label="$p_3$", linestyle="-")
        ax[2].plot(energies, params[:,9], label="$E_3$", color=line3.get_color(), linestyle="--")
        ax[2].plot(energies, params[:,10], label="$\\sigma_3$", color=line3.get_color(), linestyle=":")
        plt.show()


if __name__ == "__main__":
    main()
