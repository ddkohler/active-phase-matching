import pathlib
import matplotlib.pyplot as plt
import WrightTools as wt

here = pathlib.Path(__file__).resolve().parent

out = wt.open(here.parent / "simulation" / "empirical_phase_matching.wt5")

out.transform("L_3", "L_1", "w_1", "w_2")
out = wt.artists.interact2D(out, channel="sinc2")

plt.show()
