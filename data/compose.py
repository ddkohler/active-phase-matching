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
# create channels showing the intensity-optimized motor positions
L2_opt = np.empty((proc.w3_points.size, proc.w1_points.size))
L3_opt = np.empty((proc.w3_points.size, proc.w1_points.size))
for i, j in np.ndindex((proc.w3_points.size, proc.w1_points.size)):
    dij = proc.at(w3_points=[proc.w3_points.points[i], "wn"], w1_points=[proc.w1_points.points[j], "wn"])
    opt = np.unravel_index(np.argmax(dij.diff_norm[:]), shape=dij.diff.shape)
    L3_opt[i,j] = proc.L3_points.points[opt[0]] * 0.3/2  # convert ps to mm displacement
    L2_opt[i,j] = proc.L2_points.points[opt[1]] * 0.3/2  # convert ps to mm displacement
proc.create_channel("L1_opt", values=L2_opt[:,:,None, None], signed=True, units="mm")
proc.create_channel("L3_opt", values=L3_opt[:,:,None, None], signed=True, units="mm")

# rename variables for more transparent usage
proc.rename_variables(L2="L1", w3="w2")

c.save(here / "composed.wt5", overwrite=True)
