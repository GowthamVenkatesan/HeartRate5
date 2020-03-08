import numpy as np
import matplotlib.pyplot as plt

def fourier_series_coeff_numpy(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag

T = 1.5  # any real number
def f(t):
    """example of periodic function in [0,T]"""
    n1, n2, n3 = 1., 4., 7.  # in Hz, or nondimensional for the matter.
    a0, a1, b4, a7 = 4., 2., -1., -3
    return a0 / 2 * np.ones_like(t) + a1 * np.cos(2 * np.pi * n1 * t / T) + \
        b4 * np.sin(2 * np.pi * n2 * t / T) + a7 * np.cos(2 * np.pi * n3 * t / T)


N_chosen = 10
a0, a, b = fourier_series_coeff_numpy(f, T, N_chosen)
print(f"a[0]: {a[0]}")
print(f"a: {a}")
print(f"b: {b}")

print("done")
