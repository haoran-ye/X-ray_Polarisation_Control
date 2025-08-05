import numpy as np
from Constants import *
from scipy import fftpack, integrate, interpolate

def Gaussian(E, E0, deltaE):
  return np.exp(-(E-E0)**2 / (2 * (deltaE/2.355)**2))

def roll_zeropad(a, shift):
    """
    Roll an array by `shift` samples, padding the vacated entries with zeros.
    """
    a = np.asarray(a)
    result = np.zeros_like(a)
    if shift > 0:
        if shift < len(a):
            result[shift:] = a[:-shift]
    elif shift < 0:
        n = -shift
        if n < len(a):
            result[:-n] = a[n:]
    else:
        result = a.copy()
    return result

def convolve(f, g):
    """
    FFT based convolution

    :param f: array
    :param g: array
    :return: array, (f * g)[n]
    """
    f_fft = fftpack.fftshift(fftpack.fftn(f))
    g_fft = fftpack.fftshift(fftpack.fftn(g))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(f_fft*g_fft)))
