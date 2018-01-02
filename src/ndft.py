import numpy as np


def ndft(x, fx, k, axis=-1):
    """Non-equispaced discrete Fourier transform.

    Compute fk_j = \sum_i fx_i e^{-2 \pi i x_i k_j}.

    Save convention as numpy.fft.fft

    Examples
    --------
    >>> import numpy as np

    >>> N = 128
    >>> x = np.arange(N)
    >>> y = x
    >>> yf = np.fft.fftshift(np.fft.fft(y))
    >>> k = np.fft.fftshift(np.fft.fftfreq(N, d=x[1]-x[0]))
    >>> yf1 = ndft(x, y, k)
    >>> assert np.allclose(yf, yf1)
    >>>

    """
    assert x.ndim == 1 and x.shape[0] == fx.shape[axis], 'x does not align with axis %d of fx' % axis
    return np.tensordot(fx, np.exp(-2.0J * np.pi * np.outer(x, k)), axes=([axis, 0]))


def ndift(k, fk, x, axis=-1):
    """Non-equispaced discrete inverse Fourier transform.

    Compute fx_i = \sum_j fk_j e^{-2 \pi i k_j x_i}.

    Save convention as numpy.fft.ifft

    Examples
    --------
    >>> import numpy as np

    >>> N = 128
    >>> x = np.arange(N)
    >>> y = x
    >>> yif = np.fft.fftshift(np.fft.ifft(y))
    >>> k = np.fft.fftshift(np.fft.fftfreq(N, d=x[1]-x[0]))
    >>> yif1 = ndift(x, y, k)
    >>> assert np.allclose(yif, yif1)
    >>>
    >>> assert np.allclose(y, ndift(k, ndft(x, y, k), x))
    >>>

    """
    assert k.ndim == 1 and k.shape[0] == fk.shape[axis], 'k does not align with axis %d of fk' % axis
    return np.tensordot(fk, np.exp(2.0J * np.pi * np.outer(k, x)), axes=([axis, 0])) / fk.shape[axis]
