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


def global_fit_func(eloss, energies, r0, p0a, p0b, p0c, p0d, p0e, p1a, p1b, p1c, p1d, p1e, e1a, e1b, s1a, s1b, p2a, p2b, p2c, p2d, p2e, e2a, e2b, s2a, s2b):
    x = energies[:,np.newaxis]/energies[-1]
    u0 = np.exp(-(p0a + x*(p0b + x*(p0c + x*(p0d + x*p0e)))))
    u1 = np.exp(-(p1a + x*(p1b + x*(p1c + x*(p1d + x*p1e)))))
    u2 = np.exp(-(p2a + x*(p2b + x*(p2c + x*(p2d + x*p2e)))))
    pnorm = 1.0/(u0 + u1 + u2)
    p0 = u0*pnorm
    p1 = u1*pnorm
    p2 = u2*pnorm
    e1 = e1a + x*e1b
    s1 = s1a + x*s1b
    e2 = e2a + x*e2b
    s2 = s2a + x*s2b
    norm1 = 1.0/(s1*np.sqrt(2.0*np.pi))
    norm2 = 1.0/(s2*np.sqrt(2.0*np.pi))
    return (p0*r0*np.exp(-r0*eloss[np.newaxis,:])
        + p1*norm1*np.exp(-(eloss[np.newaxis,:] - e1)**2/(2*s1**2))
        + p2*norm2*np.exp(-(eloss[np.newaxis,:] - e2)**2/(2*s2**2)))


def opt_global_f(params, eloss, energies, histogram):
    fit_value = global_fit_func(
            eloss, energies,
            params[0],
            params[1], params[2], params[3], params[4], params[5],
            params[6], params[7], params[8], params[9], params[10],
            params[11], params[12],
            params[13], params[14],
            params[15], params[16], params[17], params[18], params[19],
            params[20], params[21],
            params[22], params[23])
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

    for filename in filenames[0:1]:
        data = np.load(filename)
        histogram = data["histogram"]
        energies = data["energies"]
        eloss = data["eloss"]
        emax = np.max(eloss)
        bounds = [
            (0.5, 100.0),
            (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0),
            (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0),
            (3.0, 8.0), (-3.0, 3.0),
            (0.01, emax), (-emax, emax),
            (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0),
            (8.0, 25.0), (-8.0, 8.0),
            (0.01, emax), (-emax, emax)
        ]

        res = opt.dual_annealing(
                opt_global_f, bounds=bounds, args=(eloss[:-1], energies[:-1], histogram), maxiter=20000)
        print(res.message)
        print(f"E0 = {1.0/res.x[0]}")
        print(f"p0 = exp(-({res.x[1]} + {res.x[2]}x + {res.x[3]}x^2 + {res.x[4]}x^3 + {res.x[5]}x^4))")
        print(f"p1 = exp(-({res.x[6]} + {res.x[7]}x + {res.x[8]}x^2 + {res.x[9]}x^3 + {res.x[10]}x^4))")
        print(f"E1 = {res.x[11]} + {res.x[12]}x")
        print(f"dE1 = {res.x[13]} + {res.x[14]}x")
        print(f"p2 = exp(-({res.x[15]} + {res.x[16]}x + {res.x[17]}x^2 + {res.x[18]}x^3 + {res.x[19]}x^4))")
        print(f"E2 = {res.x[20]} + {res.x[21]}x")
        print(f"dE2 = {res.x[22]} + {res.x[23]}x")

        # params = []
        # print(f"Processing {filename}")
        # for i in tqdm.tqdm(range(histogram.shape[0])):
        #     res = opt.shgo(opt_f, args=(eloss[:-1], histogram[i]), bounds=bounds, constraints=linear_constraints, minimizer_kwargs={"method": "COBYQA"})
        #     params.append(res.x)

        # params = np.array(params)
        histogram_fit = global_fit_func(eloss, energies, *res.x)
        # histogram_fit = np.array([[fit_func(eloss[i], *params[j]) for i in range(histogram.shape[1])] for j in range(histogram.shape[0])])

        fig, ax = plt.subplots(2)
        ax[0].imshow(histogram.T, aspect="auto")
        ax[1].imshow(histogram_fit.T, aspect="auto")

        # line0, = ax[2].plot(energies[:-1], params[:,0], label="$\\tau$", color="white", linestyle="--")
        # line1, = ax[2].plot(energies[:-1], params[:,1], label="$p_1$", linestyle="-")
        # ax[2].plot(energies[:-1], params[:,2], label="$E_1$", color=line1.get_color(), linestyle="--")
        # ax[2].plot(energies[:-1], params[:,3], label="$\\sigma_1$", color=line1.get_color(), linestyle=":")
        # line2, = ax[2].plot(energies[:-1], params[:,4], label="$p_2$", linestyle="-")
        # ax[2].plot(energies[:-1], params[:,5], label="$E_2$", color=line2.get_color(), linestyle="--")
        # ax[2].plot(energies[:-1], params[:,6], label="$\\sigma_2$", color=line2.get_color(), linestyle=":")
        # plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
