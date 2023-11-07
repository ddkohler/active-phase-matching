# conjugate DOVE phase matching
# k1 = OPA1
# k2 = OPA3
# k3 = OPA4
# k_out + k2 = k1 + k3

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize, CenteredNorm
import WrightTools as wt
import simlib as lib
import pathlib


here = pathlib.Path(__file__).resolve().parent
save = True

w1 = np.linspace(1600, 2200, 51)
w2 = np.linspace(2600, 3200, 51)
w3 = 18400

n = lib.n_caf2

# test = lib.solve_angles_boxcars(lib.initial_colors, lib.initial_angles[1]) 

# compute angles
solutions = np.empty((w1.size, w2.size, 4))
for i, wi in enumerate(w1):
    for j, wj in enumerate(w2):
        angles = lib.solve_angles_boxcars(
            [wi, wj, 18400],
            lib.initial_angles[1], 
            ri=n
        )
        solutions[i,j] = angles

# for i,j in [[0,0], [0,-1], [-1, 0], [-1,-1]]:
#     angles = solutions[i,j]
#     w1i = w1[i]
#     w2j = w2[j]
#     print(f"--- w1={w1i}, w2={w2j} ----------------------------------------------")
#     print(angles*180/np.pi)
#     print(lib.dk4([w1i, w2j, 18400], angles[:-1]))
#     for v in lib.k_vectors_internal([w1i, w2j, 18400], angles, ri=n):
#         with np.printoptions(suppress=True):
#             print(v)


sim = wt.Data(name="phase_matching_angles")
sim.create_variable("w_1", w1[:, None], units="wn")
sim.create_variable("w_2", w2[None, :], units="wn")
sim.create_variable("w_3", np.array([18400]).reshape(1,1))
for i, angle in enumerate(["alpha", "beta", "gamma", "delta"]):
    sim.create_channel(
        angle, 
        values=solutions[:,:,i] * 180 / np.pi, 
        signed=True,
        units="deg"
    )
    sim.channels[-1].null = lib.initial_angles[i] * 180 / np.pi

sim.transform("w_1", "w_2")

omega_ref = np.arcsin(1 * 25.4 / 500 / 2)  # guess value for optical axis angle

# IMPORTANT:  
# focussing mirrors have efl 500 mm, but motor displacements are half that of 
# the beam displacement
# To convert to L in terms of motor position, we use a factor of 250
# but I should really convert the experimental motors to deal with this...
sim.create_channel("L1", values=250 * np.sin(omega_ref + solutions[:,:,0]), signed=True, units="mm")
sim.L1.null = 250 * np.sin(omega_ref + lib.initial_angles[0])
sim.create_channel("L3", values=250 * np.sin(omega_ref + solutions[:,:,2]), signed=True, units="mm")
sim.L3.null = 250 * np.sin(omega_ref + lib.initial_angles[2])
sim.create_channel("k1_sine", values=np.sin(sim.alpha[:] * np.pi / 180 / n(sim.w_1[:]))*lib.kmag(sim.w_1[:]), units="wn")
sim.k1_sine.null = sim.k1_sine.min()  #  -= sim.at(w_1=[1900, "wn"], w_2=[2750, "wn"]).k1_sine[0]

temp = np.empty(sim.shape)
for ind in np.ndindex(temp.shape):
    angles = [sim[ch][:][ind] * np.pi/180 for ch in ["alpha", "beta", "gamma"]]
    colors = [sim.w_1[:][ind[0],0], sim.w_2[:][0,ind[1]], sim.w_3[:][0,0]]
    ans = lib.k_vectors_internal(colors, angles)[4]
    temp[ind] = ans[-1]
sim.create_channel("kprime", values=temp, signed=True)
sim.kprime.null = sim.at(w_1=[1900, "wn"], w_2=[2750, "wn"]).kprime[0]


for ch in sim.channel_names:
    if ch=="beta":
        continue
    wt.artists.quick2D(sim, channel=ch)
    if save:
        wt.artists.savefig(here / f"{ch}.png")


if save:
    sim.save(here / "phase_matched_solutions.wt5", overwrite=True)
else:
    plt.show()

# fig, gs = wt.artists.create_figure(cols=[1,1])
# ax = plt.subplot(gs[0])
# ax.pcolormesh(w2, w1, solutions[:,:,0]-lib.initial_angles[0], cmap=wt.artists.colormaps["signed"], norm=CenteredNorm(vcenter=0))
# ax1 = plt.subplot(gs[1])
# ax1.pcolormesh(w2, w1, solutions[:,:,2]-lib.initial_angles[2], cmap=wt.artists.colormaps["signed"], norm=CenteredNorm(vcenter=0))


# out = lib.solve_angles([1900, 2750, 18400], lib.initial_angles[-1])
# angles = [out["x"][0], out["x"][1], lib.initial_angles[-1]]
# print(out)
# print(lib.dk4([1900, 2750, 18400], angles))
# print(out["x"])
# print(lib.initial_angles)
