import pathlib
import simlib as lib
import matplotlib.pyplot as plt
import numpy as np

here = pathlib.Path(__file__).resolve().parent

wn = np.linspace(1500, 20000, 501)
plt.figure()
plt.plot(wn, lib.n_caf2(wn))
plt.title("CaF2", fontsize=36)
plt.xlabel("wavenumber (1/cm)", fontsize=26)
plt.ylabel("n", fontsize=26)
plt.xlim(wn.min(), wn.max())
plt.grid()
plt.tight_layout()
plt.savefig(here / "CaF2 refractive index.png")
