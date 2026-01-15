import glob
import itertools

import numpy as np
import matplotlib.pyplot as plt
import cycler

timesteps = [150, 190, 240, 300, 370, 580, 730, 1100, 1400]

fig, ax = plt.subplots(1, 2)
ax[0].set_title("Al")
ax[1].set_title("O")

ax[0].set_xlabel("Energy loss [eV]")
ax[1].set_xlabel("Energy loss [eV]")

ax[0].set_ylabel("Density")
ax[1].set_ylabel("Density")

bins = np.arange(0,90,5)

cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'])

for timestep in timesteps:
    al_fnames = glob.glob(f"remote_data/eloss/Al2O3/eloss_Al2O3_low_{timestep}fm_Al_*.dat")
    al_data = np.concatenate([np.loadtxt(fname) for fname in al_fnames])
    print(f"Al {timestep}: {np.median(al_data[:,3])}")
    ax[0].hist(al_data[:,3], density=True, bins=bins, label=f"{timestep} fm", facecolor="none", edgecolor=next(cycler)['color'])

cycler = itertools.cycle(plt.rcParams['axes.prop_cycle'])
for timestep in timesteps:
    o_fnames = glob.glob(f"remote_data/eloss/Al2O3/eloss_Al2O3_low_{timestep}fm_O_*.dat")
    o_data = np.concatenate([np.loadtxt(fname) for fname in o_fnames])
    print(f"O {timestep}: {np.median(o_data[:,3])}")
    ax[1].hist(o_data[:,3], density=True, bins=bins, label=f"{timestep} fm", facecolor="none", edgecolor=next(cycler)['color'])

ax[0].legend()
ax[1].legend()
plt.show()

# Al 50 as: 18.128502574312733 eV
# Al 100 as: 17.940396467827668 eV
# Al 150 as: 18.178084471452166 eV
# Al 200 as: 18.642203479699674 eV
# Al 250 as: 18.429194690652366 eV
# Al 300 as: 18.49900987860019 eV
# Al 350 as: 18.47246457210713 eV
# Al 400 as: 18.858840621018317 eV
# O 50 as: 16.75663626921596 eV
# O 100 as: 16.213071973666956 eV
# O 150 as: 17.639834294488537 eV
# O 200 as: 15.638409913604846 eV
# O 250 as: 18.16753770835203 eV
# O 300 as: 16.181012852292042 eV
# O 350 as: 17.863784771761857 eV
# O 400 as: 16.257747337134788 eV

# Al 150fm: 18.210310495422164
# Al 190fm: 17.916038651499548
# Al 240fm: 18.67825495480065
# Al 300fm: 17.61188157095603
# Al 370fm: 17.474799945346604
# Al 580fm: 18.33301826830575
# Al 730fm: 16.125676849602314
# Al 1100fm: 18.438599783345126
# Al 1400fm: 18.276933910616208
# O 150fm: 16.91505800854793
# O 190fm: 17.150492990280327
# O 240fm: 16.43925996215694
# O 300fm: 16.188641049026046
# O 370fm: 17.214956847092253
# O 580fm: 16.35464512409817
# O 730fm: 17.521034913457697
# O 1100fm: 17.753610153675254
# O 1400fm: 15.962586851652304
