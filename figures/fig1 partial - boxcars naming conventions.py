"""Not actually used in paper; figure made using ppt"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import WrightTools as wt

# create an isometric view of the boxcars geometry

here = pathlib.Path(__file__).resolve().parent
save = True

amps = [2000, 3000, 6000, 5000]
thetas = [0.5, 0.42]
thetas.append(np.arcsin(np.sin(thetas[0]) * amps[0] / amps[2]))
thetas.append(np.arcsin(np.sin(thetas[1]) * amps[1] / amps[3]))

ks = [
    np.array([np.sin(thetas[0]), 0, np.cos(thetas[0])]),
    np.array([0,np.sin(thetas[1]), np.cos(thetas[1])]),
    np.array([-np.sin(thetas[2]), 0, np.cos(thetas[2])]),
    np.array([0, -np.sin(thetas[3]), np.cos(thetas[3])]),
]


for ki, ai in zip(ks, amps):
    ki *= ai

zmax = max([k[2] for k in [ks[0]+ks[2], ks[1]+ks[3]]]) + 300
zmin = -600
xymin = 0
xymax = 1500

def to_iso(vec, theta=np.pi/4):
    x = vec[2]-np.cos(theta) * vec[0]
    y = vec[1]-np.cos(theta) * vec[0]
    return x,y

k_iso = [to_iso(v) for v in ks]


xz = [to_iso(xyz) for xyz in [[0,0,zmin],[xymax, 0, zmin],[xymax,0,zmax],[0,0,zmax]]]
yz = [to_iso(xyz) for xyz in [[0,0,zmin],[0, xymax, zmin],[0,xymax,zmax],[0,0,zmax]]]

frame = [to_iso(xyz) for xyz in [[xymax, 0, 0], [0, xymax, 0], [0,0,zmax]]]    


plt.figure()
import matplotlib.patches as patches
p_xz = patches.Polygon(xy=xz, fc="k", alpha=0.2)
p_yz = patches.Polygon(xy=yz, fc="k", alpha=0.1)

plt.gca().add_patch(p_xz)
plt.gca().add_patch(p_yz)

for v in frame:
    plt.arrow(0,0, *v, color="k")

kwargs = dict(width=40, length_includes_head=True, head_length=200)

plt.arrow(0,0, *k_iso[0], color="green", **kwargs)
plt.arrow(0,0, *k_iso[1], color="blue", **kwargs)
plt.arrow(*k_iso[0], *k_iso[2], color="green", **kwargs)
plt.arrow(*k_iso[1], *k_iso[3], color="blue", **kwargs)


if save:
    wt.artists.savefig(here / f"{pathlib.Path(__file__).name}.png")
else:
    plt.show()
