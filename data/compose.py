import pathlib
import WrightTools as wt
import numpy as np


here = pathlib.Path(__file__).resolve().parent

# in the raw data:
# OPA3 angle is static ("k2" in this work)
# l2 moves the OPA1 beamline ("k1" in this work)
# l3 moves the OPA4 beamline ("k3" in this work)
paths = dict(
    empirical = here / "w1w3l2l3_raw.wt5",
    no_pm_3mm = here / "w1w3_passive.wt5",
    l2_attenuation = here / "w1_motorcurve.wt5", 
    l3_attenuation = here / "w4_motorcurve.wt5",
    w1_power = here / "w1_powercurve.wt5",
    w3_power = here / "w3_powercurve.wt5",
)


c = wt.Collection(name="root")
c.create_collection(name="raw")

for ni, pi in paths.items():
    wt.open(pi).copy(name=ni, parent=c.raw)

#   NOTE: I cannot apply L2 and L3 attenuation corrections because the data file lists coordinates relative to the phase matched calculation.
#   In theory, I could apply the correction by converting L2 and L3 to their absolute positions using the theoretical calibration.
#   I will refrain from this for now, since the corrections are quite ad-hoc, and I am not convinced they actually account for what is happening in the sample.

proc = c.raw.empirical.copy(name="processed", parent=c)

for vi in ["w4_", "w2"]:
    proc.remove_variable(vi)
for ci in ["random_walk", "ratiow1pr", "ratiow2pr", "pyro_1", "pyro_2", "pyro_3", "pyro_4"]:
    proc.remove_channel(ci)

# power normalization channels
for data in [proc, c.raw.no_pm_3mm]:
    d_temp1 = c.raw.w1_power.map_variable("w1_points", data.w1_points[:].flatten())
    d_temp2 = c.raw.w3_power.map_variable("w3_points", data.w3_points[:].flatten())
    data.create_channel("w1_power", d_temp1.pyro_1[:].reshape(data.w1_points.shape))
    data.create_channel("w3_power", d_temp2.pyro_1[:].reshape(data.w3_points.shape))
    data.create_channel("I1I2", values=data.w1_power[:] * data.w3_power[:])
    data.create_channel("diff_norm", data.diff[:] / data.I1I2[:])
    data.smooth(2, channel="diff_norm")

proc.create_channel("pm_max", values=proc.diff_norm[:].max(axis=(2,3))[:,:,None,None])

# rename variables for more transparent usage
proc.rename_variables(L2="L1", w3="w2")

# --- --- regrid 4D dataset to l1 and l3 absolute positions ---------------------------------------
#  (instead of setpoint deviation)
offsets = wt.open(here.parent / "simulation" / "phase_matched_solutions.wt5")
for chan in ["L1", "L3"]:
    offsets[chan][:] -= offsets[chan].null
    offsets[chan].null = 0
offsets = offsets.map_variable("w_1", proc.w1_points[:].reshape(-1,1)).map_variable("w_2", proc.w2_points[:].reshape(1,-1))

proc.L1_points.convert("mm_delay")
proc.L3_points.convert("mm_delay")

l1a = np.linspace(offsets.L1.min() + proc.L1_points.min(), offsets.L1.max() + proc.L1_points.max(), 31)
l3a = np.linspace(offsets.L3.min() + proc.L3_points.min(), offsets.L3.max() + proc.L3_points.max(), 31)

d_abs = wt.Data(name="l1l3_absolute", parent=c)
d_abs.create_variable("w_1", values=offsets.w_1[:].reshape(-1,1,1,1), units="wn")
d_abs.create_variable("w_2", values=offsets.w_2[:].reshape(1,-1,1,1), units="wn")
d_abs.create_variable("L_3", values=l3a.reshape(1,1,-1,1), units="mm")
d_abs.create_variable("L_1", values=l1a.reshape(1,1,1,-1), units="mm")

interp = np.empty((d_abs.w_1.size, d_abs.w_2.size, d_abs.L_3.size, d_abs.L_1.size))

# walk w1,w2 and interpolate the offsets to the new grid
from scipy.interpolate import LinearNDInterpolator
for ij in np.ndindex((d_abs.w_1.size, d_abs.w_2.size)):
    w1i = d_abs.w_1.points[ij[0]]
    w2j = d_abs.w_2.points[ij[1]]
    print(ij, w1i, w2j)
    intensity = proc.at(w1_points=[w1i,"wn"], w2_points=[w2j,"wn"]).diff[:]
    L1s = (proc.L1_points[:][0,0] + offsets.L1[:][ij]).flatten()
    L3s = (proc.L3_points[:][0,0] + offsets.L3[:][ij]).flatten()
    points = [[L3s[k], L1s[l]] for k,l in np.ndindex((L3s.size, L1s.size))]
    values = [intensity[kl] for kl in np.ndindex((L3s.size, L1s.size))]

    interpij = LinearNDInterpolator(points, values)
    interp[ij] = interpij(l3a[:, None], l1a[None, :])

d_abs.create_channel("interp", interp)
d_abs.transform("w_1", "w_2", "L_3", "L_1")
d_abs.create_channel("I1I2", values=np.transpose(proc.I1I2[:], (1,0,2,3)))
d_abs.create_channel("interp_norm", values=d_abs.interp[:] / d_abs.I1I2[:])

c.save(here / "composed.wt5", overwrite=True)
