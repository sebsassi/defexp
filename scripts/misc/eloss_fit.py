import argparse
import os
import glob

import tqdm
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def fit_func(eloss, r0, p1, e1, s1, p2, e2, s2):
    p0 = 1.0 - p1 - p2
    norm1 = 1.0/(s1*np.sqrt(2.0*np.pi))
    norm2 = 1.0/(s2*np.sqrt(2.0*np.pi))
    return (p0*r0*np.exp(-r0*eloss)
            + p1*norm1*np.exp(-(eloss - e1)**2/(2*s1**2))
            + p2*norm2*np.exp(-(eloss - e2)**2/(2*s2**2)))


def fit_func_der(eloss, r0, p1, e1, s1, p2, e2, s2):
    p0 = 1.0 - p1 - p2
    norm1 = 1.0/(s1*np.sqrt(2.0*np.pi))
    norm2 = 1.0/(s2*np.sqrt(2.0*np.pi))
    return np.array([
        p0*(1 - r0*eloss)*np.exp(-r0*eloss),
        norm1*np.exp(-(eloss - e1)**2/(2*s1**2)) - np.exp(-r0*eloss),
        p1*norm1*np.exp(-(eloss - e1)**2/(2*s1**2))*(eloss - e1)/s1**2,
        p1*norm1*np.exp(-(eloss - e1)**2/(2*s1**2))*((eloss - e1)**2 - s1**2)/s1**3,
        norm2*np.exp(-(eloss - e2)**2/(2*s2**2)) - np.exp(-r0*eloss),
        p2*norm2*np.exp(-(eloss - e2)**2/(2*s2**2))*(eloss - e2)/s2**2,
        p2*norm2*np.exp(-(eloss - e2)**2/(2*s2**2))*((eloss - e2)**2 - s2**2)/s2**3,
    ])


def opt_f_df(params, eloss, histogram):

    fit_value = fit_func(
            eloss,
            params[0],
            params[1], params[2], params[3],
            params[4], params[5], params[6])
    f = (1.0/histogram.size)*np.sum((fit_value - histogram)**2)

    d_fit = fit_func_der(
            eloss,
            params[0],
            params[1], params[2], params[3],
            params[4], params[5], params[6])
    df = (2.0/histogram.size)*np.sum((fit_value - histogram)*d_fit, axis=-1)

    return f, df


def opt_f(params, eloss, histogram):

    fit_value = fit_func(
            eloss,
            params[0],
            params[1], params[2], params[3],
            params[4], params[5], params[6])
    f = (1.0/histogram.size)*np.sum((fit_value - histogram)**2)

    return f


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
        bounds = [
            (0.0, np.inf),
            (0.0, 1.0),
            (3.0, 8.0),
            (0.01, emax),
            (0.0, 1.0),
            (8.0, 25.0),
            (0.01, emax)
        ]

        ineq_constr = {
            "type": "ineq",
            "fun": lambda x: np.array([
                1.0 - x[1] - x[4] - x[7],
                x[5] - x[2],
                x[8] - x[5]]),
            "jac": lambda x: np.array([
                [0.0, -1.0,  0.0, 0.0, -1.0,  0.0, 0.0],
                [0.0,  0.0, -1.0, 0.0,  0.0,  1.0, 0.0],
                [0.0,  0.0,  0.0, 0.0,  0.0, -1.0, 0.0]])
        }

        constraint_matrix = np.array([
            [0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0,  1.0, 0.0]])

        constraint_bound_low = np.array([0.0, 0.0])
        constraint_bound_high = np.array([1.0, np.inf])

        linear_constraints = opt.LinearConstraint(
                constraint_matrix, constraint_bound_low, constraint_bound_high)

        params = []
        print(f"Processing {filename}")
        for i in tqdm.tqdm(range(histogram.shape[0])):
            # res = opt.shgo(opt_f_df, args=(eloss, histogram[i]), bounds=bounds, constraints=ineq_constr, options={"jac": True})
            res = opt.shgo(opt_f, args=(eloss, histogram[i]), bounds=bounds, constraints=linear_constraints, minimizer_kwargs={"method": "COBYQA"})
            params.append(res.x)

        params = np.array(params)
        histogram_fit = np.array([[fit_func(eloss[i], *params[j]) for i in range(histogram.shape[1])] for j in range(histogram.shape[0])])

        fig, ax = plt.subplots(3)
        ax[0].imshow(histogram.T, aspect="auto")
        ax[1].imshow(histogram_fit.T, aspect="auto")

        line0, = ax[2].plot(energies, params[:,0], label="$\\tau$", color="white", linestyle="--")
        line1, = ax[2].plot(energies, params[:,1], label="$p_1$", linestyle="-")
        ax[2].plot(energies, params[:,2], label="$E_1$", color=line1.get_color(), linestyle="--")
        ax[2].plot(energies, params[:,3], label="$\\sigma_1$", color=line1.get_color(), linestyle=":")
        line2, = ax[2].plot(energies, params[:,4], label="$p_2$", linestyle="-")
        ax[2].plot(energies, params[:,5], label="$E_2$", color=line2.get_color(), linestyle="--")
        ax[2].plot(energies, params[:,6], label="$\\sigma_2$", color=line2.get_color(), linestyle=":")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
