import argparse

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, nargs='+', help="thermo file to plot")
args = parser.parse_args()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for filename in args.filename:
    data = np.load(filename)
    thermo = data["thermo"]

    ax1.plot(thermo[:,0], thermo[:,1])
    # ax2.plot(thermo[:,0], thermo[:,2])

plt.show()
