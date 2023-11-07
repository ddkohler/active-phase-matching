# conjugate DOVE phase matching
# k1 = OPA1
# k2 = OPA3
# k3 = OPA4
# k_out + k2 = k1 + k3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import WrightTools as wt
import simlib as lib
import pathlib


here = pathlib.Path(__file__).resolve().parent
save = True

ell = 0.25  # pathlength (cm)

w1 = np.linspace(1600, 2200, 101)
w2 = np.linspace(2600, 3200, 101)

reference = lib.initial_angles

angles = list(reference)
angles[2] += 0.007  # similar to Emily's offset?

mismatch1 = np.empty((w1.size, w2.size))
mismatch2 = mismatch1.copy()
for i, wi in enumerate(w1):
    for j, wj in enumerate(w2):
        mismatch1[i,j] = lib.dk4([wi, wj, 18400], angles)
        mismatch2[i,j] = lib.dk4([wi, wj, 18400], reference)

# save simulated output for comparison with experimental
out = wt.Data()
out.create_variable("w1", values=w1[:, None], units="wn")
out.create_variable("w2", values=w2[None, :], units="wn")
out.create_channel("phase_mismatch_init", values=mismatch2, signed=True)  # note this is radians/cm
out.create_channel("phase_mismatch_alt", values=mismatch1, signed=True)  # note this is radians/cm
out.create_channel("sinc2_init", values=lib.sinc(mismatch2*ell / 2)**2)
out.create_channel("sinc2_alt", values=lib.sinc(mismatch1*ell / 2)**2)
out.transform("w1", "w2")

fig, gs = wt.artists.create_figure(cols=[1,1])
for i, chan in enumerate(["sinc2_init", "sinc2_alt"]):
    axi = plt.subplot(gs[i])
    axi.pcolormesh(out, channel=chan, norm=Normalize(vmin=0), cmap=wt.artists.colormaps["default"])
    wt.artists.set_ax_labels(axi, xlabel=out.axes[0].label, ylabel=out.axes[1].label)
    axi.grid(True, color="k", linestyle=":")
    axi.set_title(chan)
if save:
    wt.artists.savefig(here / "phase_mismatch_passive.png")
else:
    plt.show()

if save:
    out.save(here / "phase_mismatch_passive.wt5", overwrite=True)
