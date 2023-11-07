import pathlib
import WrightTools as wt
import matplotlib.pyplot as plt

here = pathlib.Path(__file__).resolve().parent

d_abs = wt.open(here.parent / "data" / "composed.wt5").l1l3_absolute
d_abs.print_tree()

out3 = wt.artists.interact2D(d_abs, channel="interp_norm")
out4 = wt.artists.interact2D(d_abs, xaxis="L_1", yaxis="L_3")

plt.show()
