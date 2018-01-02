import os
from collections import defaultdict
import numpy as np
from numpy.linalg import matrix_rank
import h5py
from astropy.cosmology import Planck13 as cosmo
from rpca import ialm
from ndft import ndft, ndift
import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



conv_beam = config.conv_beam
D = config.D

if conv_beam:
    in_dir = '../results/flat_sky/conv_%.1f/' % D
    out_dir = '../results/check_flat/conv_%.1f/' % D
else:
    in_dir = '../results/flat_sky/no_conv/'
    out_dir = '../results/check_flat/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# read in multi-frequency map of the 21 cm signal
with h5py.File(in_dir+'cm.hdf5', 'r') as f:
    nf, ny, nx = f['map'].shape
    lonra = f['map'].attrs['lonra'].tolist()
    latra = f['map'].attrs['latra'].tolist()
    cm = f['map'][:]

freqs = np.linspace(700.0, 800.0, nf) # MHz
freq0 = 1420.4 # MHz
zs = freq0 / freqs - 1.0 # redshifts
# get comoving distance
cd = cosmo.comoving_distance(zs).value # Mpc
# print cosmo.h
cd /= cosmo.h # Mpc / h
# get k_parallel by approximate cd as uniform
k_paras = np.fft.fftshift(2*np.pi * np.fft.fftfreq(nf, d=(cd[0]-cd[-1])/nf)) # h Mpc^-1
k_paras = k_paras[nf/2:] # get only positive k_paras


# 3D inverse Fourier transform of cm to get its Fourier modes
# NOTE: approximate the line of sight distances as uniform
cmk3 = np.fft.fftshift((nf*nx*ny) * np.fft.ifftn(cm))
# array to save the 2D Fourier transform in transverse plans of cm
cmk2 = np.zeros_like(cm, dtype=np.complex128)
# array to save kx and ky
kxs = np.zeros((nf, nx)) # h Mpc^-1
kys = np.zeros((nf, ny)) # h Mpc^-1
for fi, z in enumerate(zs):
    # 2D inverse Fourier transform for each z
    cmk2[fi] = np.fft.fftshift((nx * ny) * np.fft.ifft2(cm[fi]))
    # compute kx, ky
    lon_mpc = 1.0e-3 * cosmo.kpc_proper_per_arcmin(z) * (lonra[1] - lonra[0]) * 60
    lon_mpch = lon_mpc / cosmo.h # Mpc / h
    lat_mpc = 1.0e-3 * cosmo.kpc_proper_per_arcmin(z) * (latra[1] - latra[0]) * 60
    lat_mpch = lat_mpc / cosmo.h # Mpc / h
    kxs[fi] = np.fft.fftshift(2*np.pi * np.fft.fftfreq(nx, d=lon_mpch/nx)) # h Mpc^-1
    kys[fi] = np.fft.fftshift(2*np.pi * np.fft.fftfreq(ny, d=lat_mpch/ny)) # h Mpc^-1

# use only central frequency kx, ky to approximate all freqs
kxs = kxs[nf/2]
kys = kys[nf/2]


factor = 0.5
kpbin = int(factor * np.sqrt((ny/2.0)**2 + (nx/2.0)**2)) # bin for k_perp
# only use the central freq ks to bin
k_bins = np.linspace(0, (kpbin+2)/(kpbin+1)*np.sqrt(kxs[0]**2 + kys[0]**2), kpbin+1)
k_perps = np.array([ (k_bins[i] + k_bins[i+1])/2 for i in range(kpbin) ])
# print k_perps
# get the corresponds kx, ky in each bin
kpmodes = defaultdict(list)
for yi in range(ny):
    for xi in range(nx):
        # get the bin index
        bi = np.searchsorted(k_bins, np.sqrt(kxs[xi]**2 + kys[yi]**2))
        # drop (0, 0) mode
        if bi == 0:
            continue
        kpmodes[bi-1].append((yi, xi))


Pk3 = (cmk3 * cmk3.conj()).real # K, the true Pk(kz, ky, kx) for comparison
Pk2 = np.zeros((kpbin, nf)) # K, to save true Pk(k_perp, k_para)
Pkk = np.zeros((kpbin, nf, nf)) # K, to save all Pkk
Pkkd = np.zeros((kpbin, nf)) # K, to save diagonal of all Pkk
for bi in range(kpbin):
    # print bi, len(kpmodes[bi])
    Tk2 = np.zeros((nf, len(kpmodes[bi])), dtype=cmk2.dtype)
    Tk3 = np.zeros((nf, len(kpmodes[bi])), dtype=cmk3.dtype)
    for i, (yi, xi) in enumerate(kpmodes[bi]):
        Pk2[bi] += Pk3[:, yi, xi]
        Tk2[:, i] = cmk2[:, yi, xi]
        Tk3[:, i] = cmk3[:, yi, xi]
    Pk2[bi] /= len(kpmodes[bi])

    # compute freq covariance matrix of this bin
    corr2 = np.dot(Tk2, Tk2.T.conj()) / len(kpmodes[bi])

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(corr2.real, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(corr2.imag, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.savefig(out_dir + 'corr2zz_%d.png' % bi)
    # plt.close()


    # compute and check Pk(k_perp, k_para, k'_para)
    corr2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(nf * np.fft.ifft(corr2, axis=0), axes=0), axis=1), axes=1)

    corr3 = np.dot(Tk3, Tk3.T.conj()) / len(kpmodes[bi])

    assert np.allclose(corr2, corr3)

    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(corr2.real, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(222)
    # plt.imshow(corr2.imag, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(223)
    # plt.imshow(corr3.real, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(corr3.imag, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.savefig(out_dir + 'corr2_and_corr3_%d.png' % bi)
    # plt.close()


    # plt.figure()
    # plt.subplot(211)
    # plt.plot(np.diag(corr2.real))
    # plt.subplot(212)
    # plt.plot(np.diag(corr2.imag))
    # plt.savefig(out_dir + 'corr2_and_corr3_diag_%d.png' % bi)
    # plt.close()

    Pkk[bi] = corr2.real
    Pkkd[bi] = np.diag(corr2.real)

# check
assert np.allclose(Pkkd, Pk2)

# plot Pkkd and Pk2
plt.figure()
plt.subplot(121)
# times 1000 to mK
# plt.imshow(1000 * Pkkd.T, origin='lower', aspect='auto', vmax=100)
im = 1000 * Pkkd.T[nf/2:, :] # mK
m, n = im.shape
plt.pcolormesh(im, vmax=100)
plt.xlim(0, n)
plt.ylim(0, m)
plt.colorbar()
plt.subplot(122)
plt.imshow(im, origin='lower', aspect='auto', interpolation='nearest', vmax=100)
plt.colorbar()
plt.savefig(out_dir + 'Pkkd.png')
plt.close()


# bin to get Pk
kbin = int(factor * np.sqrt((nf/2.0)**2 + (ny/2.0)**2 + (nx/2.0)**2))
k_bins = np.linspace(np.sqrt(k_paras[0]**2 + k_perps[0]**2), (kbin+2)/(kbin+1)*np.sqrt(k_paras[-1]**2 + k_perps[-1]**2), kbin+1)
ks = np.array([ (k_bins[i] + k_bins[i+1])/2 for i in range(kbin) ])
# print ks
# get the corresponds k_paras, k_perp in each bin
kmodes = defaultdict(list)
for yi in range(kpbin): # for perp
    for xi in range(nf/2): # for para
        # drop 0 mode of k_paras
        if xi == 0:
            continue
        # get the bin index
        bi = np.searchsorted(ks, np.sqrt(k_perps[yi]**2 + k_paras[xi]**2))
        kmodes[bi].append((yi, xi))


Pk = np.zeros((kbin,)) # K, to save all Pk
for bi in range(kbin):
    # print bi, len(kmodes[bi])
    for i, (yi, xi) in enumerate(kmodes[bi]):
        Pk[bi] += Pkkd[yi, xi+nf/2]
    Pk[bi] /= len(kmodes[bi])

# plog Pk
plt.figure()
plt.loglog(ks, 1000 * ks**3 * Pk / (2 * np.pi**2))
plt.xlabel(r'$k \ (h \, \rm{Mpc}^{-1})$')
plt.ylabel(r'$\Delta (k)^2 \ (\rm{mK})$')
plt.savefig(out_dir + 'Pk.png')
plt.close()


# bin input 21cm to get Pk
Pk_input = np.zeros((kbin,)) # to save all input Pk
for bi in range(kbin):
    modes = []
    cnt = 0
    for zi in range(nf/2, nf):
        for yi in range(ny/2, ny):
            for xi in range(nx/2, nx):
                k = np.sqrt(k_paras[zi-nf/2]**2 + kys[yi]**2 + kxs[xi]**2)
                if k_bins[bi] <= k and k < k_bins[bi+1]:
                    cnt += 1
                    Pk_input[bi] += Pk3[zi, yi, xi]
    Pk_input[bi] /= cnt

# check Pk and Pk_input
# plog Pk and Pk_input
plt.figure()
plt.loglog(ks, 1000 * ks**3 * Pk / (2 * np.pi**2), label='Pk')
plt.loglog(ks, 1000 * ks**3 * Pk_input / (2 * np.pi**2), label='Pk_input')
plt.xlabel(r'$k \ (h \, \rm{Mpc}^{-1})$')
plt.ylabel(r'$\Delta (k)^2 \ (\rm{mK})$')
plt.legend()
plt.savefig(out_dir + 'Pk_check.png')
plt.close()
