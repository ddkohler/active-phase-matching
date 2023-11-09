import pathlib
import WrightTools as wt
import matplotlib.pyplot as plt
import numpy as np

here = pathlib.Path(__file__).resolve().parent
save = True

exp = wt.open(here.parent / "data" / "composed.wt5").processed
exp.transform("w1_points", "w2_points", "L1_points", "L3_points")
# exp.print_tree()

passive = wt.open(here.parent / "data" / "composed.wt5").raw.no_pm_3mm
# reassign var names to match manuscript names
passive.remove_variable("w2")
passive.rename_variables(L2="L1", w3="w2")

passive.smooth(2, channel="I1I2")
passive.transform("w1_points", "w2_points")
# passive.print_tree()

sim = wt.open(here.parent / "simulation" / "phase_mismatch_passive.wt5")
# sim.print_tree()

fig, gs = wt.artists.create_figure(
    width="single", cols=[1,1,"cbar"], nrows=2, hspace=0.6
)

ax0 = plt.subplot(gs[0,0])
ax0.set_title(r"$I_\mathsf{out}$")
plt.xticks(visible=False)
ax1 = plt.subplot(gs[0,1])
ax1.set_title(r"$I_\mathsf{out} / I_1I_2$")
plt.yticks(visible=False)
plt.xticks(visible=False)
ax2 = plt.subplot(gs[1,0])
ax2.set_title(r"$\mathsf{Initial \ Simulation}$")
ax3 = plt.subplot(gs[1,1])
ax3.set_title(r"$\mathsf{Adjusted \ Angles}$")
plt.yticks(visible=False)

passive.smooth(2, channel="diff")
ax0.pcolormesh(passive, channel="diff")

ax1.pcolormesh(passive, channel="diff_norm")
levels = np.linspace(-40,40, 5)
cs = ax1.contour(sim, channel="phase_mismatch_alt", levels=levels, colors=["k" if li !=0 else "white" for li in levels]) #, cmap=wt.artists.colormaps["signed"])
fmt = lambda x: r"$\mathsf{" + f"{np.abs(x):.0f}" + r"}\ \mathsf{cm}^{-1}$"
ax1.clabel(cs, cs.levels, inline=True, fontsize=12, fmt=fmt)

ax2.pcolormesh(sim, channel="sinc2_init")
ax3.pcolormesh(sim, channel="sinc2_alt")

cax = plt.subplot(gs[:,2])
wt.artists.plot_colorbar(cax, label=r"$\mathsf{Signal \ Intensity \ (norm.)}$", ticklocation="right", label_fontsize=18)

for ax in [ax0, ax2]:
    wt.artists.set_ax_labels(ax, ylabel=r"$\bar{\nu}_2 \ (\mathsf{cm}^{-1})$")
for ax in [ax2, ax3]:
    wt.artists.set_ax_labels(ax, xlabel=r"$\bar{\nu}_1 \ (\mathsf{cm}^{-1})$")

for i, ax in enumerate(fig.axes):
    if ax != cax:
        ax.grid(visible=True, color="k", lw=0.5, linestyle=":")
        wt.artists.corner_text("abcd"[i], ax=ax, background_alpha=0.5)
        if ax != ax3:
            ax.scatter(1900, 2750, s=100, c="goldenrod")

if not save:
    plt.show()
else:
    wt.artists.savefig(here / f"{pathlib.Path(__file__).name[:-3]}.png")
