import pathlib
import WrightTools as wt
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import numpy as np

save = True
here = pathlib.Path(__file__).resolve().parent
L1_label = r"$\Delta L_1 \ (\mathsf{mm})$"
L3_label = r"$\Delta L_3 \ (\mathsf{mm})$"
w1_label = r"$\bar{\nu}_1 \ (\mathsf{cm}^{-1})$"
w2_label = r"$\bar{\nu}_2 \ (\mathsf{cm}^{-1})$"

wn_ref = [1900, 2750]  # wn
# these are the colors we used as targets for alignment

exp = wt.open(here.parent / "data" / "composed.wt5").processed
exp.transform("w1_points", "w2_points", "L1_points", "L3_points")
exp.L1_points.convert("mm_delay")
exp.L3_points.convert("mm_delay")

sim = wt.open(here.parent / "simulation" / "phase_matched_solutions.wt5")
sim4d = wt.open(here.parent / "simulation" / "empirical_phase_matching.wt5")
sim4d.transform("w_1", "w_2", "L_1", "L_3")

fig, gs = wt.artists.create_figure(
    width="single", cols=[1,1], nrows=5, hspace=0.1,
    aspects=[
        [[0,0], 0.1],
        [[1,0], 1],
        [[2,0], 0.6],
        [[3,0], 0.1],
        [[4,0], 1]
    ]
)

ax0 = plt.subplot(gs[4,0])
ax1 = plt.subplot(gs[4,1])
plt.yticks(visible=False)
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[1,1])
plt.yticks(visible=False)

cax1 = plt.subplot(gs[3,:])
cax2 = plt.subplot(gs[0,0])
cax3 = plt.subplot(gs[0,1])

wt.artists.plot_colorbar(
    cax=cax1,
    orientation="horizontal",
    ticklocation="top",
    label=r"$\mathsf{Signal \ Intensity \ (norm.)}$",
)

exp1 = exp.at(w1_points=[wn_ref[0], "wn"], w2_points=[wn_ref[1], "wn"])
ax0.pcolormesh(exp1, channel="diff_norm")

ax1.pcolormesh(
    sim4d.at(w_1=[wn_ref[0], "wn"], w_2=[wn_ref[1], "wn"]),
    channel="sinc2"
)

sim.transform("w_1", "w_2")
sim.L1[:] -= sim.at(w_1=[wn_ref[0], "wn"], w_2=[wn_ref[1], "wn"]).L1[0]
sim.L1.null = 0

sim.L3[:] -= sim.at(w_1=[wn_ref[0], "wn"], w_2=[wn_ref[1], "wn"]).L3[0]
sim.L3.null = 0

pixels = ax2.pcolormesh(sim, channel="L1")
cbar2 = plt.colorbar(
    pixels,
    cax=cax2,
    orientation="horizontal",
    ticklocation="top",
    norm=CenteredNorm(halfrange=np.abs(sim.L1[:]).max()),
    cmap="signed"
)
cbar2.set_label(r"$L_1 \ \mathsf{setpoint \ (mm)}$", fontsize=18)

pixels = ax3.pcolormesh(sim, channel="L3")
cbar3 = plt.colorbar(
    pixels,
    cax=cax3,
    orientation="horizontal",
    ticklocation="top",
    norm=CenteredNorm(halfrange=np.abs(sim.L3[:]).max()),
    cmap="signed"
)
cbar3.set_label(r"$L_3 \ \mathsf{setpoint \ (mm)}$", fontsize=18)

# link calibrations to our specific investigation
for ax in [ax2, ax3]:
    ax.scatter(wn_ref[0], wn_ref[1], s=100, c="goldenrod")
for ax in [ax0, ax1]:
    wt.artists.set_ax_spines(ax, c="goldenrod")

wt.artists.set_ax_labels(ax0, ylabel=L3_label)
wt.artists.set_ax_labels(ax2, ylabel=w2_label)
for ax in [ax0, ax1]:
    wt.artists.set_ax_labels(ax, xlabel=L1_label)
for ax in [ax2, ax3]:
    wt.artists.set_ax_labels(ax, xlabel=w1_label)

for i, ax in enumerate([ax2, ax3, ax0, ax1]):
    if ax not in [cax1, cax2, cax3]:
        ax.grid(visible=True, color="k", lw=0.5, linestyle=":")
        wt.artists.corner_text("abcd"[i], ax=ax, background_alpha=0.5)

if not save:
    plt.show()
else:
    wt.artists.savefig(here / f"{pathlib.Path(__file__).name}.png")

