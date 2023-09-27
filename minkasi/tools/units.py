import numpy as np

kb = 1.38064852e-16
h = 6.62607004e-27
T = 2.725


def y2rj(freq: float = 90) -> float:
    """
    Get the y to Rayleigh-Jeans conversion at a given frequency.

    Parameters
    ----------
    freq : float, default: 90
        The frequency to get the conversion at.

    Returns
    -------
    y2rj : float
        Conversion to multiply a y map by to get a Rayleigh-Jeans normalized map note
        that it doesn't have the T_cmb at the end, so the value for low frequencies is -2.
    ."""
    x = freq * 1e9 * h / kb / T
    ex = np.exp(x)
    f = x**2 * ex / (ex - 1) ** 2 * (x * (ex + 1) / (ex - 1) - 4)
    return f


def planck_g(freq: float = 90) -> float:
    """
    Conversion between T_CMB and T_RJ as a function of frequency.

    Parameters
    ----------
    freq : float, default: 90
        The frequency to get the conversion at.

    Returns
    -------
    planck_g : float
        The conversion factor.
    """
    x = freq * 1e9 * h / kb / T
    ex = np.exp(x)
    return x**2 * ex / ((ex - 1) ** 2)
