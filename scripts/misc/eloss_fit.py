import argparse

import numpy as np
import scipy.optimize as opt


def fit_func(eloss, p0, r0, p1, e1, s1, p2, e2, s2, p3, e3, s3):
    return (p0*np.exp(-r0*eloss)
            + p1*np.exp(-(eloss - e1)**2/(2*s1**2))
            + p2*np.exp(-(eloss - e2)**2/(2*s2**2))
            + p3*np.exp(-(eloss - e3)**2/(2*s3**2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="file containing 2d histogram")
    args = parser.parse_args()

    data = np.load(filename)
    histogram = data["histogram"]
    energies = data["energies"]
    eloss = data["eloss"]

    pi = 0.25
    bounds = ([0,   0,      0,  0,       0,      0,     0,      0,      0,  0,      0       ],
              [1,   np.inf, 1,  np.inf,  np.inf, 1,     np.inf, np.inf, 1,  np.inf, np.inf  ])
    init =    [pi,  1.0,    pi, 5.0,     1.0,    pi,    10.0,   1.0     pi, 15.0,   1.0]

    params = []
    for i in range(histogram.shape[1]):
        popt, pcov = opt.curve_fit(fit_func, eloss, histogram[:,i], p0=init, bounds=bounds)
        params.append(popt)
        init = popt

    params = np.array(params)
    histogram_fit = np.array([fit_func(eloss[i], params[i]) for i in range(histogram.shape[1])]).T

    fig, ax = plt.subplots(3)
    ax[0].imshow(histogram)
    ax[1].imshow(histogram_fit)

    line0 = ax[3].plot(energies, params[:,0], label="$p_0$", linestyle="-")
    ax[3].plot(energies, params[:,1], label="$\\tau$", color=line0.get_color(), linestyle="--")
    line1 = ax[3].plot(energies, params[:,2], label="$p_1$", linestyle="-")
    ax[3].plot(energies, params[:,3], label="$E_1$", color=line1.get_color(), linestyle="--")
    ax[3].plot(energies, params[:,4], label="$\\sigma_1$", colro=line1.get_color(), linestyle=":")
    line2 = ax[3].plot(energies, params[:,5], label="$p_2$", linestyle="-")
    ax[3].plot(energies, params[:,6], label="$E_2$", color=line2.get_color(), linestyle="--")
    ax[3].plot(energies, params[:,7], label="$\\sigma_2$", color=line2.get_color(), linestyle=":")
    line3 = ax[3].plot(energies, params[:,8], label="$p_3$", linestyle="-")
    ax[3].plot(energies, params[:,9], label="$E_3$", color=line3.get_color(), linestyle="--")
    ax[3].plot(energies, params[:,10], label="$\\sigma_3$", color=line3.get_color, linestyle=":")


if __name__ == "__main__":
    main()
