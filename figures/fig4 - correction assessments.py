import pathlib
import WrightTools as wt
import matplotlib.pyplot as plt
import numpy as np

here = pathlib.Path(__file__).resolve().parent
save = False

exp = wt.open(here.parent / "data" / "composed.wt5").processed
exp.smooth(2, channel="diff")
exp.transform("w1_points", "w2_points", "L1_points", "L3_points")
# exp.print_tree()

fig, gs = wt.artists.create_figure(
    width="single", cols=[1,1,"cbar"], nrows=2, hspace=0.1
)

ax0 = plt.subplot(gs[0,0])
ax0.set_title(r"$\mathsf{Calculated}$")
plt.xticks(visible=False)
ax1 = plt.subplot(gs[0,1])
ax1.set_title(r"$\mathsf{Empirical}$")
plt.yticks(visible=False)
plt.xticks(visible=False)
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[1,1])
plt.yticks(visible=False)

cax = plt.subplot(gs[:,2])
wt.artists.plot_colorbar(cax, label=r"$\mathsf{Signal \ Intensity \ (norm.)}$", ticklocation="right", label_fontsize=18)

d_boxcars = exp.at(
    # L1_points=[1.5, "mm_delay"], L3_points=[0, "mm_delay"]
    L1_points=[0, "mm_delay"], L3_points=[0, "mm_delay"]
)
d_boxcars.transform("w1_points", "w2_points")

exp.create_channel("diff_max", exp.diff[:].max(axis=(2,3)).reshape(exp.pm_max.shape))
d_empirical = exp.at(L3_points=[0,"ps"], L1_points=[0, "ps"])  # coords do not matter
d_empirical.transform("w1_points", "w2_points")

if False:
    common_mag1 = max(d_boxcars.diff[:].max(), d_empirical.diff_max[:].max())
    ax0.pcolormesh(d_boxcars, channel="diff", vmax=common_mag1)
    ax1.pcolormesh(d_empirical, channel="diff_max", vmax=common_mag1)

    common_mag2 = max(d_boxcars.diff_norm[:].max(), d_empirical.pm_max[:].max())
    ax2.pcolormesh(d_boxcars, channel="diff_norm", vmax=common_mag2)
    ax3.pcolormesh(d_empirical, channel="pm_max", vmax=common_mag2)
else:
    ax0.pcolormesh(d_boxcars, channel="diff")
    ax1.pcolormesh(d_empirical, channel="diff_max")
    ax2.pcolormesh(d_boxcars, channel="diff_norm")
    ax3.pcolormesh(d_empirical, channel="pm_max")


for ax in [ax0, ax2]:
    wt.artists.set_ax_labels(ax, ylabel=r"$\bar{\nu}_2 \ (\mathsf{cm}^{-1})$")
for ax in [ax2, ax3]:
    wt.artists.set_ax_labels(ax, xlabel=r"$\bar{\nu}_1 \ (\mathsf{cm}^{-1})$")

for i, ax in enumerate(fig.axes):
    if ax != cax:
        ax.grid(visible=True, color="k", lw=0.5, linestyle=":")
        wt.artists.corner_text("abcd"[i], ax=ax, background_alpha=0.5)

if not save:
    plt.show()
else:
    wt.artists.savefig(here / f"{pathlib.Path(__file__).name[:-3]}.png")
