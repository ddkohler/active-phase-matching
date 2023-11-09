# conjugate DOVE phase matching
# k1 = OPA1 (the redder IR)
# k2 = OPA3 (the bluer IR)
# k3 = OPA4
# k_out + k2 = k1 + k3
# k_out is redder than k3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import WrightTools as wt
import simlib as lib
import pathlib

here = pathlib.Path(__file__).resolve().parent

ell = 0.3  # pathlength (cm)

solution = lib.solve_angles(lib.initial_colors, lib.initial_angles[-1])["x"]

dangle = np.linspace(-0.1, 0.1)
angle1 = dangle - solution[0]
angle2 = dangle - solution[1]
# scan about solution
dks = np.empty((dangle.size, dangle.size))
for ind in np.ndindex(dks.shape):
    i, j = ind
    angles = [
        dangle[i]-solution[0], 
        dangle[j]-solution[1],
        lib.initial_angles[-1]
    ]
    dks[i,j] = lib.dk4(lib.initial_colors, angles)



# Using Emily's initial parameters to check my calculation

# save simulated output for comparison with experimental
out = wt.Data()
out.create_variable("alpha", values=angle1[:, None], units="wn")
out.create_variable("beta", values=angle2[None, :], units="wn")
out.create_channel("dk4", values=dks, signed=True)
out.transform("alpha", "beta")

if True:  # examine simulation
    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.pcolormesh(out, cmap=wt.artists.colormaps["signed"])
    plt.show()

#   the answer is NO; if we allow k1 and k3 to vary within the yz plane, 
#   then we can find solutions where k4 does not lie in the xz plane (i.e. non-BOXCARS)
#   these are perfectly fine solutions from an experimental standpoint, however



