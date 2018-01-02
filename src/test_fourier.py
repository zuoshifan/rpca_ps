import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


t = np.linspace(0, 4*np.pi, 200)
x = np.sin(t)

# plt.figure()
# plt.plot(x)
# plt.savefig('x.png')
# plt.close()

xk = np.fft.fftshift(np.fft.fft(x))

# plt.figure()
# plt.plot(xk.real)
# plt.plot(xk.imag)
# plt.savefig('xk.png')
# plt.close()

xx = np.outer(x, x.conj())
xxk = np.fft.fftshift(200 * np.fft.ifft(np.fft.fftshift(np.fft.fft(xx, axis=0), axes=0), axis=1), axes=1)
xkxk = np.outer(xk, xk.conj())

print np.allclose(xxk, xkxk)

# plt.figure()
# plt.subplot(221)
# plt.imshow(xxk.real)
# plt.colorbar()
# plt.subplot(222)
# plt.imshow(xxk.imag)
# plt.colorbar()
# plt.subplot(223)
# plt.imshow(xkxk.real)
# plt.colorbar()
# plt.subplot(224)
# plt.imshow(xkxk.imag)
# plt.colorbar()
# plt.savefig('xxk.png')
# plt.close()

y = np.cos(2*t)
X = np.c_[x, y]
# print X.shape

Xk = np.fft.fftshift(np.fft.fft(X, axis=0), axes=0)

XX = np.dot(X, X.T.conj())
XXk = np.fft.fftshift(200 * np.fft.ifft(np.fft.fftshift(np.fft.fft(XX, axis=0), axes=0), axis=1), axes=1)
XkXk = np.dot(Xk, Xk.T.conj())

print np.allclose(XXk, XkXk)
