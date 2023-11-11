"""
"""

import pathlib
import WrightTools as wt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec


here = pathlib.Path(__file__).resolve().parent
save = True

d = wt.open(here.parent / "data" / "composed.wt5").processed
# d.print_tree()

# fig, gs0 = wt.artists.create_figure(width="dissertation")

d1 = d[:,:,0, 0]
# d1.print_tree()
d1.transform("w1_points", "w2_points")
fig, gs = wt.artists.create_figure(cols=[1,"cbar"])
ax1 = fig.add_subplot(gs[0])
ax1.pcolormesh(d1, channel="I1I2")

sax = wt.artists.add_sideplot(ax1, along="x", pad=0.05)
say = wt.artists.add_sideplot(ax1, along="y", pad=0.05)
y0 = d1.I1I2[:].mean(axis=0)
y0 /= y0.max()
y1 = d1.I1I2[:].mean(axis=1)
y1 /= y1.max()

sax.fill_between(d1.w1_points.points, y0, color="k", alpha=0.4)
say.fill_betweenx(d1.w2_points.points, y1, color="k", alpha=0.4)

cax = fig.add_subplot(gs[1])
wt.artists.plot_colorbar(cax, label=r"$\mathsf{Signal} \ (\mathsf{norm.})$", label_fontsize=18)
wt.artists.set_ax_labels(
    ax1,
    xlabel=r"$\bar{\nu}_1 \ (\mathsf{cm}^{-1})$",
    ylabel=r"$\bar{\nu}_1 \ (\mathsf{cm}^{-1})$"
)

ax1.set_xlim(d.w1_points.min(), d.w1_points.max())
ax1.set_ylim(d.w2_points.min(), d.w2_points.max())

wt.artists.corner_text(r"$I_1I_2$", ax=ax1)
wt.artists.corner_text(r"$I_1$", ax=sax)
wt.artists.corner_text(r"$I_2$", ax=say)

if save:
    wt.artists.savefig(here / f"{pathlib.Path(__file__).name[:-3]}.png")
else:
    plt.show()
