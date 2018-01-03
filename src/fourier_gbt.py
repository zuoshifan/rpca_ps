import os
from collections import defaultdict
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
from scipy.signal import argrelmax
from astropy.cosmology import Planck13 as cosmo
import h5py
from spca import decompose
import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# which data set to use
field = '15hr'
section = 'A'
mask = True # whether mask the boarders

# clean map file name
if field == '15hr':
    mp_fl = '/project/ycli/data/gbt/15hr/sec%s_15hr_41-80_pointcorr_clean_map_I_800.npy' % section
elif field == '1hr':
    mp_fl = '/project/ycli/data/gbt/1hr/sec%s_1hr_80-28_ptcorr_clean_map_I_800.npy' % section
else:
    raise ValueError('Unknown field %s' % field)

freq_start = 0
freq_end = None
# freq_start = 25
# # freq_end = 251
# freq_end = 205

if not mask:
    ra_start = 0
    ra_end = None
    dec_start = 0
    dec_end = None
else:
    # mask boarder
    if field == '15hr':
        ra_start = 10
        ra_end = 68
        dec_start = 5
        dec_end = 38
    elif field == '1hr':
        ra_start = 20
        ra_end = 141
        dec_start = 15
        dec_end = 68


out_dir = '../results/gbt/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# read in meta info
f = open(mp_fl+'.meta')
meta = f.readline()
f.close()
exec('meta = %s' % meta)
# print meta
# print type(meta)


# read in the multi-frequency map
mp = np.load(mp_fl)
print mp.shape
nf, nra, ndec = mp.shape

# # plot map
# fis = range(10) + range(120, 130) + range(250, nf)
# for fi in fis:
#     plt.plot()
#     plt.imshow(mp[fi], origin='lower', aspect='auto')
#     plt.colorbar()
#     plt.savefig(out_dir + 'map_%d.png' % fi)
#     plt.close()

# mp_sum = mp.reshape(nf, -1).sum(axis=1)
# med = np.median(mp_sum)
# abs_diff = np.abs(mp_sum - med)
# mad = np.median(abs_diff) / 0.6745
# inds = np.where(abs_diff > 2.5*mad)[0]
# print inds
# # print mp_sum[inds]
# # # plot mp_sum
# # plt.figure()
# # plt.plot(mp_sum)
# # plt.plot(inds, mp_sum[inds], 'ro')
# # plt.savefig(out_dir + 'mp_sum.png')
# # plt.close()
# mp[inds] = 0

# make all axes in ascending order
mp = mp[::-1, ::-1, :]

# generate the corresponding freqs, ras, decs
freqs = np.arange(-nf/2, nf/2)*meta['freq_delta'] + meta['freq_centre'] # Hz
freqs = freqs[::-1] # in ascending order
freqs *= 1.0e-6 # MHz
ras = np.arange(-nra/2, nra/2)*meta['ra_delta'] + meta['ra_centre'] # degree
ras = ras[::-1] # in ascending order
decs = np.arange(-ndec/2, ndec/2)*meta['dec_delta'] + meta['dec_centre'] # degree, ascending order


rs, re = ra_start, ra_end
ds, de = dec_start, dec_end
fs, fe = freq_start, freq_end
mp = mp[fs:fe, rs:re, ds:de] # mask boarder
mp = np.transpose(mp, (0, 2, 1)) # (freq, dec, ra)

nf, ny, nx = mp.shape

# for fi in range(10) + [nf/2] + range(246, 256):
#     plt.figure()
#     # plt.imshow(mp[nf/2].T, origin='lower', aspect='auto')
#     plt.imshow(mp[fi].T, origin='lower', aspect='auto')
#     plt.colorbar()
#     plt.savefig(out_dir + 'mp_%d.png' % fi)
#     plt.close()

# get the cropped axes
ras = ras[rs:re]
lonra = [ras[0], ras[-1]] # lon range
decs = decs[ds:de]
latra = [decs[0], decs[-1]] # lat range
print lonra, latra

freqs = freqs[fs:fe]
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


# 3D inverse Fourier transform of map to get its Fourier modes
# NOTE: approximate the line of sight distances as uniform
# array to save the 2D Fourier transform in transverse plans of sky map
mapk2 = np.zeros_like(mp, dtype=np.complex128)
# array to save kx and ky
kxs = np.zeros((nf, nx)) # h Mpc^-1
kys = np.zeros((nf, ny)) # h Mpc^-1
for fi, z in enumerate(zs):
    # 2D inverse Fourier transform for each z
    mapk2[fi] = np.fft.fftshift((nx * ny) * np.fft.ifft2(mp[fi]))
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

# print k_paras
# print kxs[nx/2:]
# print kys[ny/2:]

factor = 0.6
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


# print ny, nx
# for bi in range(kpbin):
#     print bi, len(kpmodes[bi])

Pkk = np.zeros((kpbin, nf, nf)) # K, to save all Pkk of the extracted cm
Pkkd = np.zeros((kpbin, nf)) # K, to save diagonal of all Pkk of the extracted cm
for bi in range(kpbin):
# for bi in [8, 9, 10]:
    nkp = len(kpmodes[bi])
    # print bi, nkp
    mapkp2 = np.zeros((nf, nkp), dtype=mapk2.dtype)
    for i, (yi, xi) in enumerate(kpmodes[bi]):
        mapkp2[:, i] = mapk2[:, yi, xi]

    # compute freq covariance matrix of this bin
    map_corr = np.dot(mapkp2, mapkp2.T.conj()) / nkp

    # plot map_corr
    plt.figure()
    plt.subplot(121)
    plt.imshow(map_corr.real, origin='lower', aspect='equal')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(map_corr.imag, origin='lower', aspect='equal')
    plt.colorbar()
    plt.savefig(out_dir + 'corr_%d.png' % bi)
    plt.close()


    # RPCA decomposition map_corr
    # L, S = ialm(map_corr.real, tol1=1.0e-9, tol2=1.0e-7, max_iter=2000)
    # L, S = ialm(map_corr, tol1=1.0e-9, tol2=1.0e-7, max_iter=2000, verbose=True)

    # SPCA decomposition map_corr
    rank = min(nkp, 50)
    L, S = decompose(map_corr.real, rank=rank, tol=1.0e-14, max_iter=1000)

    # if nkp <= 5:
    #     L, S = decompose(map_corr.real, rank=nkp, tol=1.0e-14, max_iter=1000)
    # else:
    #     L, S = decompose(map_corr.real, rank=10, tol=1.0e-14, max_iter=1000)
    #     # print matrix_rank(L), matrix_rank(S)
    #     U, s, VT = la.svd(L)
    #     # print np.diff(np.log10(s[:20]))
    #     # print argrelmax(np.diff(np.log10(s[:20])))
    #     rank = argrelmax(np.diff(np.log10(s[:20])))[0][0] + 1
    #     # rank = min(6, rank)
    #     rank += 20
    #     L, S = decompose(map_corr.real, rank=rank, tol=1.0e-12, max_iter=1000)
    print matrix_rank(L), matrix_rank(S)

    # # clean S
    # di = np.diag_indices(nf) # diag indeices of S
    # for ii in range(nf):
    #     if ii == 0:
    #         dis = [ di, ]
    #     else:
    #         di1 = ( (di[0]+ii)[:-ii], di[1][:-ii] )
    #         di2 = ( di[0][:-ii], (di[1]+ii)[:-ii] )
    #         dis = [ di1, di2 ]
    #     for dix in dis:
    #         if ii < 20:
    #             d = S[dix]
    #             # med = np.median(d)
    #             # abs_diff = np.abs(d - med)
    #             # mad = np.median(abs_diff) / 0.6745
    #             # print len(d), med, mad
    #             # S[dix] = np.where(abs_diff>3.0*mad, 2.5*mad, d)
    #             mu = np.mean(d)
    #             std = np.std(d)
    #             print len(d), mu, std
    #             S[dix] = np.where(np.abs(d-mu)>3.0*std, 0, d)
    #         else:
    #             S[dix] = 0


    # plot decomp
    M = map_corr.real
    plt.figure()
    plt.subplot(221)
    plt.imshow(M, origin='lower', aspect='equal')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(M-L-S, origin='lower', aspect='equal')
    # plt.imshow(cm_corr.real-S, origin='lower', aspect='equal')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(L, origin='lower', aspect='equal')
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(S, origin='lower', aspect='equal')
    plt.colorbar()
    plt.savefig(out_dir + 'decomp_%d.png' % bi)
    plt.close()

    # compute and check Pk(k_perp, k_para, k'_para)
    Sf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(nf * np.fft.ifft(S, axis=0), axes=0), axis=1), axes=1)

    Pkk[bi] = Sf.real
    Pkkd[bi] = np.diag(Sf.real)

# plot Pkkd and Pkkd_input
plt.figure()
im = 1000 * Pkkd.T[nf/2:, :] # mK
plt.imshow(im, origin='lower', aspect='auto', interpolation='nearest', vmin=0)
# plt.imshow(im, origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=10000)
plt.colorbar()
plt.savefig(out_dir + 'Pkkd_decomp.png')
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


Pk = np.zeros((kbin,)) # K, to save all Pk of the extracted cm
for bi in range(kbin):
    # print bi, len(kmodes[bi])
    for i, (yi, xi) in enumerate(kmodes[bi]):
        Pk[bi] += Pkkd[yi, xi+nf/2]
    if len(kmodes[bi]) == 0:
        Pk[bi] == np.nan
    else:
        Pk[bi] /= len(kmodes[bi])

# get inds where Pk > 0
inds = np.where(Pk>0)[0]

# plog Pk
plt.figure()
# plt.loglog(ks, 1000 * ks**3 * Pk / (2 * np.pi**2))
plt.loglog(ks[inds], 1000 * ks[inds]**3 * Pk[inds] / (2 * np.pi**2))
plt.xlabel(r'$k \ (h \, \rm{Mpc}^{-1})$')
plt.ylabel(r'$\Delta (k)^2 \ (\rm{mK})$')
plt.savefig(out_dir + 'Pk_decomp.png')
plt.close()
