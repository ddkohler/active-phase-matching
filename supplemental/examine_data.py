import pathlib
import WrightTools as wt
import matplotlib.pyplot as plt

here = pathlib.Path(__file__).resolve().parent

c = wt.open(here.parent / "data" / "composed.wt5")
d1 = c.processed
# all corrections used the pyro_1 channel
d2 = c.raw.l2_attenuation
d3 = c.raw.l3_attenuation
d4 = c.raw.w1_power  
d5 = c.raw.w3_power

# map power curves to my points and normalize
# from scipy.interpolate import PchipInterpolator
d1.L2_points.convert("mm_delay")
d1.L3_points.convert("mm_delay")
d2.L2_points.convert("mm_delay")
d3.L3_points.convert("mm_delay")

if True:
    out = wt.artists.interact2D(d1, channel="diff_norm", xaxis="L3_points", yaxis="L2_points")
    out2 = wt.artists.interact2D(d1, channel="diff", xaxis="w1_points", yaxis="w3_points")

if False:
    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.plot(d2, channel="pyro_1")
    ax.plot(d3, channel="pyro_1")

if False:  # compare boxcars with empirical
    d_boxcars = d1.at(L2_points=[1.2, "mm_delay"], L3_points=[-0.3, "mm_delay"])
    d_boxcars.transform("w1_points", "w3_points")
    wt.artists.quick2D(d_boxcars, channel="diff_norm")
    d_empirical = d1.at(L3_points=[0,"ps"], L2_points=[0, "ps"])
    d_empirical.transform("w1_points", "w3_points")
    wt.artists.quick2D(d_empirical, channel="pm_max")

if False:
    d_empirical = d1.at(L3_points=[0,"ps"], L2_points=[0, "ps"])
    d_empirical.transform("w1_points", "w3_points")
    print(d_empirical.L2_opt.null)
    wt.artists.quick2D(d_empirical, channel="L2_opt")
    wt.artists.quick2D(d_empirical, channel="L3_opt")


if False:
    import matplotlib.pyplot as plt

    # fig, gs = wt.artists.create_figure()
    # ax = plt.subplot(gs[0])
    # ax.plot(d2, channel="pyro_1")
    # ax.plot(d3, channel="pyro_1")

    # fig, gs = wt.artists.create_figure()
    # ax = plt.subplot(gs[0])
    # d4.pyro_1.normalize()
    # d5.pyro_1.normalize()
    # ax.plot(d4, channel="pyro_1")
    # ax.plot(d5, channel="pyro_1")

    plt.figure()
    # plt.pcolormesh(
    #     d4.w1_points[:], 
    #     d5.w3_points[:], 
    #     d4.pyro_1[:][None,:]*d5.pyro_1[:][:,None], 
    #     vmin=0, cmap=wt.artists.colormaps["default"]
    # )
    plt.pcolormesh(
        d_temp1.w1_points[:], 
        d_temp2.w3_points[:], 
        d_temp1.pyro_1[:][None,:]*d_temp2.pyro_1[:][:,None], 
        vmin=0, cmap=wt.artists.colormaps["default"]
    )


plt.show()
