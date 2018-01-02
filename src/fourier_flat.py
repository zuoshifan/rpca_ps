import os
from collections import defaultdict
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
from scipy.signal import argrelmax
import h5py
from astropy.cosmology import Planck13 as cosmo
from rpca import ialm
from spca import decompose
from ndft import ndft, ndift
import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



conv_beam = config.conv_beam
D = config.D

if conv_beam:
    in_dir = '../results/flat_sky/conv_%.1f/' % D
    out_dir = '../results/pk_flat/conv_%.1f/' % D
else:
    in_dir = '../results/flat_sky/no_conv/'
    out_dir = '../results/pk_flat/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# read in multi-frequency sky maps
with h5py.File(in_dir+'ps.hdf5', 'r') as f:
    nf, ny, nx = f['map'].shape
    lonra = f['map'].attrs['lonra'].tolist()
    latra = f['map'].attrs['latra'].tolist()
    ps = f['map'][:]
with h5py.File(in_dir+'ga.hdf5', 'r') as f:
    ga = f['map'][:]
with h5py.File(in_dir+'cm.hdf5', 'r') as f:
    cm = f['map'][:]

freqs = np.linspace(700.0, 800.0, nf) # MHz
freq0 = 1420.4 # MHz
zs = freq0 / freqs - 1.0 # redshifts
# get comoving distance
cd = cosmo.comoving_distance(zs).value # Mpc
# print cosmo.h
cd /= cosmo.h # Mpc / h
# N = 200
# # set appropriate k_parallel
# kp = np.logspace(-2.0, -0.3, N) # h Mpc^-1
# get k_parallel by approximate cd as uniform
k_paras = np.fft.fftshift(2*np.pi * np.fft.fftfreq(nf, d=(cd[0]-cd[-1])/nf)) # h Mpc^-1
k_paras = k_paras[nf/2:] # get only positive k_paras

# # plot cd
# plt.figure()
# plt.plot(cd)
# plt.plot([0, len(cd)], [cd[0], cd[-1]], 'k--')
# plt.savefig(out_dir + 'cd.png')
# plt.close()


# 3D inverse Fourier transform of map to get its Fourier modes
# NOTE: approximate the line of sight distances as uniform
# array to save the 2D Fourier transform in transverse plans of sky map
mapk2 = np.zeros_like(ps, dtype=np.complex128)
# array to save the 2D Fourier transform in transverse plans of cm
cmk2 = np.zeros_like(cm, dtype=np.complex128)
# array to save kx and ky
kxs = np.zeros((nf, nx)) # h Mpc^-1
kys = np.zeros((nf, ny)) # h Mpc^-1
for fi, z in enumerate(zs):
    # 2D inverse Fourier transform for each z
    mapk2[fi] = np.fft.fftshift((nx * ny) * np.fft.ifft2(ps[fi]+ga[fi]+cm[fi]))
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


Pkk_input = np.zeros((kpbin, nf, nf)) # K, to save all Pkk of input cm
Pkkd_input = np.zeros((kpbin, nf)) # K, to save diagonal of all Pkk of input cm
Pkk = np.zeros((kpbin, nf, nf)) # K, to save all Pkk of the extracted cm
Pkkd = np.zeros((kpbin, nf)) # K, to save diagonal of all Pkk of the extracted cm
for bi in range(kpbin):
    nkp = len(kpmodes[bi])
    # print bi, nkp
    mapkp2 = np.zeros((nf, nkp), dtype=mapk2.dtype)
    cmkp2 = np.zeros((nf, nkp), dtype=cmk2.dtype)
    for i, (yi, xi) in enumerate(kpmodes[bi]):
        mapkp2[:, i] = mapk2[:, yi, xi]
        cmkp2[:, i] = cmk2[:, yi, xi]

    # compute freq covariance matrix of this bin
    map_corr = np.dot(mapkp2, mapkp2.T.conj()) / nkp
    cm_corr = np.dot(cmkp2, cmkp2.T.conj()) / nkp


    # # plot map_corr and cm_corr
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(map_corr.real, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(222)
    # plt.imshow(map_corr.imag, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(223)
    # plt.imshow(cm_corr.real, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(cm_corr.imag, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.savefig(out_dir + 'corr_%d.png' % bi)
    # plt.close()


    # RPCA decomposition map_corr
    # L, S = ialm(map_corr.real, tol1=1.0e-9, tol2=1.0e-7, max_iter=2000)
    # L, S = ialm(map_corr, tol1=1.0e-9, tol2=1.0e-7, max_iter=2000, verbose=True)

    # SPCA decomposition map_corr
    if nkp <= 5:
        L, S = decompose(map_corr.real, rank=nkp, tol=1.0e-14)
    else:
        L, S = decompose(map_corr.real, rank=10, tol=1.0e-14)
        # print matrix_rank(L), matrix_rank(S)
        U, s, VT = la.svd(L)
        # print np.diff(np.log10(s[:20]))
        # print argrelmax(np.diff(np.log10(s[:20])))
        rank = argrelmax(np.diff(np.log10(s[:20])))[0][0] + 1
        # rank = min(6, rank)
        L, S = decompose(map_corr.real, rank=rank, tol=1.0e-12)
    print matrix_rank(L), matrix_rank(S)

    # # plot decomp
    # M = map_corr.real
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(M, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(222)
    # # plt.imshow(M-L-S, origin='lower', aspect='equal')
    # plt.imshow(cm_corr.real-S, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(223)
    # plt.imshow(L, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(S, origin='lower', aspect='equal')
    # plt.colorbar()
    # plt.savefig(out_dir + 'decomp_%d.png' % bi)
    # plt.close()

    # compute and check Pk(k_perp, k_para, k'_para)
    Sf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(nf * np.fft.ifft(S, axis=0), axes=0), axis=1), axes=1)
    # cmf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(nf * np.fft.ifft(cm_corr, axis=0), axes=0), axis=1), axes=1)
    cmf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(nf * np.fft.ifft(cm_corr.real, axis=0), axes=0), axis=1), axes=1)

    Pkk[bi] = Sf.real
    Pkkd[bi] = np.diag(Sf.real)
    Pkk_input[bi] = cmf.real
    Pkkd_input[bi] = np.diag(cmf.real)

# # plot Pkkd and Pkkd_input
# plt.figure()
# plt.subplot(121)
# # times 1000 to mK
# im = 1000 * Pkkd_input.T[nf/2:, :] # mK
# plt.imshow(im, origin='lower', aspect='auto', interpolation='nearest', vmax=100)
# plt.colorbar()
# plt.subplot(122)
# im1 = 1000 * Pkkd.T[nf/2:, :] # mK
# plt.imshow(im, origin='lower', aspect='auto', interpolation='nearest', vmax=100)
# plt.colorbar()
# plt.savefig(out_dir + 'Pkkd_decomp.png')
# plt.close()


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


Pk_input = np.zeros((kbin,)) # K, to save all Pk of the input cm
Pk = np.zeros((kbin,)) # K, to save all Pk of the extracted cm
for bi in range(kbin):
    # print bi, len(kmodes[bi])
    for i, (yi, xi) in enumerate(kmodes[bi]):
        Pk[bi] += Pkkd[yi, xi+nf/2]
        Pk_input[bi] += Pkkd_input[yi, xi+nf/2]
    if len(kmodes[bi]) == 0:
        Pk[bi] == np.nan
        Pk_input[bi] == np.nan
    else:
        Pk[bi] /= len(kmodes[bi])
        Pk_input[bi] /= len(kmodes[bi])

# plog Pk
plt.figure()
plt.loglog(ks, 1000 * ks**3 * Pk_input / (2 * np.pi**2), label='input')
plt.loglog(ks, 1000 * ks**3 * Pk / (2 * np.pi**2), label='recovered')
plt.legend()
plt.xlabel(r'$k \ (h \, \rm{Mpc}^{-1})$')
plt.ylabel(r'$\Delta (k)^2 \ (\rm{mK})$')
plt.savefig(out_dir + 'Pk_decomp.png')
plt.close()
