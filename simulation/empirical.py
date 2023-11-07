# 4D simulation of phase matching

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import WrightTools as wt
import simlib as lib
import pathlib

here = pathlib.Path(__file__).resolve().parent
wt5_path = here / "empirical_phase_matching.wt5"

ell = 0.3  # pathlength (cm)

w1 = np.linspace(1600, 2200, 16)
w2 = np.linspace(2600, 3200, 16)
L1 = np.linspace(-5, 5)
L3 = np.linspace(-1.8, 1.8)

gammas = np.linspace(-0.2, -1.5) * np.pi / 180
alphas = np.linspace(4, 12) * np.pi / 180

omega_ref = np.arcsin(2 * 25.4 / 250)


out = wt.Data()
out.create_variable("w_1", values=w1.reshape(-1,1,1,1), units="wn")
out.create_variable("w_2", values=w2.reshape(1,-1,1,1), units="wn")
out.create_variable("L_1", values=L1.reshape(1,1,-1,1), units="mm")
out.create_variable("L_3", values=L3.reshape(1,1,1,-1), units="mm")
# out.create_variable("alpha", values=alphas.reshape(1,1,-1,1), units="rad")
# out.create_variable("gamma", values=gammas.reshape(1,1,1,-1), units="rad")

alpha_pm = np.empty((w1.size, w2.size, 1, 1))
gamma_pm = np.empty(alpha_pm.shape)
delta_pm = np.empty(alpha_pm.shape)
mismatch = np.empty((w1.size, w2.size, L1.size, L3.size))
for ind in np.ndindex(mismatch.shape):
    w1i = w1[ind[0]]
    w2i = w2[ind[1]]
    pm = lib.solve_angles_boxcars([w1i, w2i, 18400], lib.initial_angles[1])
    alphai = pm[0] + np.arcsin(L1[ind[2]] / 250)
    gammai = pm[2] + np.arcsin(L3[ind[3]] / 250)
    alpha_pm[ind[0], ind[1]] = pm[0]
    gamma_pm[ind[0], ind[1]] = pm[2]
    delta_pm[ind[0], ind[1]] = pm[3]    
    mismatch[ind] = lib.dk4([w1i, w2i, 18400], [alphai, lib.initial_angles[1], gammai])

# save simulated output for comparison with experimental
out.create_channel("alpha_pm", values=alpha_pm)
out.create_channel("gamma_pm", values=gamma_pm)
out.create_channel("delta_pm", values=delta_pm)
out.create_channel("phase_mismatch", values=mismatch, signed=True)  # note this is radians/cm
out.create_channel("sinc2", values=lib.sinc(mismatch*ell / 2)**2)

out.transform("w_1", "w_2", "L_1", "L_3")
out.save(wt5_path, overwrite=True)



