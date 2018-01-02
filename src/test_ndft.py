import numpy as np
from ndft import ndft, ndift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# N = 128
# t = np.linspace(-4, 4, N)
# # y = np.sin(2*np.pi*t) # + np.random.randn(N)
# y = np.exp(2.0J*np.pi*t) # + np.random.randn(N)

# yf = np.fft.fftshift(np.fft.fft(y))
# k = np.fft.fftshift(np.fft.fftfreq(N, d=t[1]-t[0]))

# yf1 = ndft(t, y, k)

# ---------------------------------------------------
N = 128
t = np.arange(N)
y = t

yf = np.fft.fftshift(np.fft.fft(y))
yif = np.fft.fftshift(np.fft.ifft(y))
k = np.fft.fftshift(np.fft.fftfreq(N, d=t[1]-t[0]))

yf1 = ndft(t, y, k)
yif1 = ndift(t, y, k)

print np.allclose(yf, yf1)
print np.allclose(yif, yif1)

print np.allclose(y, np.fft.ifft(np.fft.fft(y)))
print np.allclose(y, ndift(k, ndft(t, y, k), t))


# plt.figure()
# plt.subplot(311)
# plt.plot(t, y.real)
# plt.plot(t, y.imag)
# plt.subplot(312)
# plt.plot(k, yf.real, label='1')
# plt.plot(k, yf1.real, label='2')
# plt.legend()
# plt.subplot(313)
# plt.plot(k, yf.imag, label='1')
# plt.plot(k, yf1.imag, label='2')
# plt.legend()
# plt.savefig('test_ndft.png')
# plt.close()

plt.figure()
plt.subplot(311)
plt.plot(t, y.real)
plt.plot(t, y.imag)
plt.subplot(312)
plt.plot(k, yif.real, label='1')
plt.plot(k, yif1.real, label='2')
plt.legend()
plt.subplot(313)
plt.plot(k, yif.imag, label='1')
plt.plot(k, yif1.imag, label='2')
plt.legend()
plt.savefig('test_ndft.png')
plt.close()
