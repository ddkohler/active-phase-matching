import pathlib
import WrightTools as wt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec


here = pathlib.Path(__file__).resolve().parent
save = True
channel=2

d = wt.open(here.parent / "data" / "composed.wt5").l1l3_absolute
# d.print_tree()

fig, gs0 = wt.artists.create_figure(width="dissertation",  cols=[1,"cbar"], wspace=0.5)
cax = fig.add_subplot(gs0[1])
gs = gridspec.GridSpecFromSubplotSpec(4,4, subplot_spec=gs0[0], wspace=0.05, hspace=0.05)

axes = []
w1s = sorted(d.w_1[:].flatten()[::5])
w2s = sorted(d.w_2[:].flatten()[::5])
norm = Normalize(vmax=d.channels[channel].max() * 0.7)

for i, j in np.ndindex((4, 4)):
    axij = plt.subplot(gs[3-i,j])
    axes.append(axij)
    dij = d.at(w_1=[w1s[j], "wn"], w_2=[w2s[i], "wn"])
    dij.transform("L_1", "L_3")
    axij.pcolormesh(dij, channel=channel, norm=norm)
    best_point = np.unravel_index(np.nanargmax(dij.channels[channel][:]), shape=dij.shape)
    axij.scatter(dij.L_1[0,best_point[1]], dij.L_3[best_point[0], 0], marker="*", color="k", s=100)
    axij.text(
        0, d.L_3.min(),
        f"({dij.constants[0].value:.0f}, {dij.constants[1].value:.0f})", 
        verticalalignment="bottom",
        horizontalalignment="center"
    )
    axij.grid(True, ls=":", lw=0.5, color="k")

    if i > 0: plt.xticks(visible=False)
    if j > 0: plt.yticks(visible=False)

wt.artists.plot_colorbar(cax, label=r"$I_\mathsf{out} / I_1 I_2$", ticklocation="right", label_fontsize=24)
fig.text(0.45, 0.05, r"$L_1 \ (\mathsf{mm})$", fontsize=24, horizontalalignment="center", verticalalignment="top")
fig.text(0.04, 0.5, r"$L_3 \ (\mathsf{mm})$", fontsize=24, horizontalalignment="right", verticalalignment="center", rotation="vertical")


if save:
    wt.artists.savefig(here / f"{pathlib.Path(__file__).name[:-3]}.png")
else:
    plt.show()
