import numpy as np
from scipy.interpolate import PchipInterpolator
import pathlib


# __all__ = [
#     "k4",
#     "dk4",
#     "kmag",
#     "n_caf2",
#     "k_vectors_internal"
# ]


here = pathlib.Path(__file__).resolve().parent
initial_colors = [1900, 2750, 18400]


def sinc(x):
    # different defintion of sinc than numpy (factor of pi)
    return np.sinc(x / np.pi) * np.pi


def _get_caf2_index_function():
    wavelength, n = np.loadtxt(here / "Malitson.csv", skiprows=1, unpack=True, delimiter=",")
    return PchipInterpolator(1e4 / wavelength[::-1], n[::-1])


n_caf2 = _get_caf2_index_function()


def kmag(wn):
    """
    calculate the length of the k-vector in units of wavenumbers 
    |k| = n * w / c = n * wn * 2 * pi
    """
    return n_caf2(wn) * wn * 2 * np.pi


def k_vectors_internal(wns, angles, ri=n_caf2):
    sins = [np.sin(angles[i]) / ri(wi) for i, wi in enumerate(wns)]
    coss = [(1-sini**2)**0.5 for sini in sins]
    k1 = np.array([sins[0], 0 , coss[0]])
    k2 = np.array([0, sins[1], coss[1]])
    k3 = np.array([sins[2], 0, coss[2]])
    k1 *= kmag(wns[0])
    k2 *= kmag(wns[1])
    k3 *= kmag(wns[2])
    k4 = k1-k2+k3
    return k1, k2, k3, k4, k1+k3, k2+k4


def k4(wns, angles):
    """
    parameters
    ----------
    wns : list-like
        list of colors (w1, w2, w3, w4); use positive frequencies
    angles : list like
        input angles of beams in air (radians); refraction is taken into account in calculation
    """
    ns = [n_caf2(wni) for wni in wns]
    # account for refraction
    refracted = [np.arcsin(np.sin(angles[i]) / n_caf2(wns[i])) for i in range(len(wns))]
    ss = [kmag(wns[i]) * np.sin(angle) for i, angle in enumerate(refracted)]
    cs = [kmag(wns[i]) * np.cos(angle) for i, angle in enumerate(refracted)]
    k4 = np.array([
        ss[0] + ss[2],
        -ss[1],
        cs[0] - cs[1] + cs[2]
    ])
    return k4


def dk4(wns, angles):
    w4 = wns[0] - wns[1] + wns[2]
    pol = (k4(wns, angles)**2).sum()
    pol **= 0.5
    return pol - kmag(w4)


# def solve_angles(wns, angle3):
#     """ constrain k4 to lie in the plane normal to k1-k3 plane (i.e. BOXCARS)
#     """
#     from scipy.optimize import least_squares 
#     def func(x, wns=wns, angle3=angle3):
#         angle1, angle2 = x
#         return dk4(wns, [angle1, angle2, angle3])
#     result = least_squares(func, x0=np.array(initial_angles)[:-1])
#     return result
    

def solve_angles_boxcars(wns, beta, ri=n_caf2):
    """ Emily's method
    """
    w4 = wns[0] - wns[1] + wns[2]
    k1_mag = kmag(wns[0])
    k2_mag = kmag(wns[1])
    k3_mag = kmag(wns[2])
    k4_mag = kmag(w4)

    sin2 = np.sin(beta) / ri(wns[1])
    cos2 = (1-sin2**2)**0.5

    # k4 and k2 off-axis contributions cancel
    sin4 = -k2_mag / k4_mag * sin2
    cos4 = (1-sin4**2)**0.5

    # law of cosines to solve for alpha
    kprime = k4_mag * cos4 + k2_mag * cos2
    cos1 = (k1_mag**2 + kprime**2 - k3_mag**2) / (2 * k1_mag * kprime)
    sin1 = (1-cos1**2)**0.5

    # k3 and k1 off-axis contributions cancel
    sin3 = -k1_mag / k3_mag * sin1

    # alert if the boxcars cannot be solved
    if (cos1**2 >= 1) or (sin3**2 >= 1) or (sin4**2 >= 1):
        print(sin1, sin2, sin3, sin4)
        return np.nan, np.nan, np.nan, np.nan

    # law of refraction to translate to angles in air
    sin1 *= ri(wns[0])
    sin3 *= ri(wns[2])
    sin4 *= ri(w4)
    
    return np.arcsin(sin1), beta, np.arcsin(sin3), np.arcsin(sin4)


initial_angles = solve_angles_boxcars(initial_colors, 5.39 * np.pi / 180 * n_caf2(initial_colors[1]))

