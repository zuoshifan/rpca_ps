import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
from scipy.signal import argrelmax
import h5py
import healpy
from cora.util import hputil
from rpca import ialm
from spca import decompose
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


conv_beam = config.conv_beam
D = config.D

if conv_beam:
    # map_dir = '../results/conv_beam/conv_%.1f/' % D
    # cm_name = 'smooth_21cm_256_700_800_256.hdf5'
    in_dir = '../results/alm/conv_%.1f/' % D
    out_dir = '../results/cl/conv_%.1f/' % D
else:
    # map_dir = '../sky_map/'
    # cm_name = 'sim_21cm_256_700_800_256.hdf5'
    in_dir = '../results/alm/no_conv/'
    out_dir = '../results/cl/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

ps_alm_name = 'alm_pointsource_256_700_800_256.hdf5'
ga_alm_name = 'alm_galaxy_256_700_800_256.hdf5'
cm_alm_name = 'alm_21cm_256_700_800_256.hdf5'

# with h5py.File(map_dir+cm_name, 'r') as f:
#     if len(f['map'].shape) == 3:
#         cm_map = f['map'][:, 0, :]
#     else:
#         cm_map = f['map'][:, :]
# load data from files
with h5py.File(in_dir+ps_alm_name, 'r') as f:
    ps_alm = f['alm'][:]
with h5py.File(in_dir+ga_alm_name, 'r') as f:
    ga_alm = f['alm'][:]
with h5py.File(in_dir+cm_alm_name, 'r') as f:
    cm_alm = f['alm'][:]

fg_alm = ps_alm + ga_alm # foreground
tt_alm = fg_alm + cm_alm # total signal

# # plot alm
# plt.figure()
# # plt.subplot(121)
# # plt.imshow(tt_alm[128].T.real, origin='lower', aspect='equal')
# plt.imshow(np.abs(tt_alm[128].T), origin='lower', aspect='equal', vmax=0.1)
# plt.xlabel('$l$')
# plt.ylabel('$m$')
# plt.colorbar()
# # plt.subplot(122)
# # plt.imshow(tt_alm[128].T.imag, origin='lower', aspect='equal')
# # plt.colorbar()
# plt.savefig(outdir + 'alm.png')
# plt.close()

def corr(alm, l):
    # calculate a_l0(\nu) a^*_l0(\nu')
    l0 = np.outer(alm[:, l, 0], alm[:, l, 0]).real
    # calculate correaltion matrix
    # \sum_m a_lm(\nu) a^*_lm(\nu') = 2 * \sum_{m>=0} Re{a_lm(\nu) a^*_lm(\nu')} - a_l0(\nu) a^*_l0(\nu')
    cl = (2 * np.dot(alm[:, l], alm[:, l].T.conj()).real - l0) / (2*l + 1)

    return cl

cv = tt_alm.shape[0] / 2 # central freq index
ls = []
cls_input = []
cls = []
lmax = tt_alm.shape[1]
for l in range(0, lmax, 1):
    print 'l = %d' % l
    cm_cm_corr = corr(cm_alm, l)
    tt_tt_corr = corr(tt_alm, l)

    # RPCA decompose tt_tt_corr
    # L, S = ialm(tt_tt_corr, tol1=1.0e-14, tol2=1.0e-5, max_iter=2000)
    # L, S = ialm(tt_tt_corr, tol1=1.0e-10, tol2=1.0e-8, max_iter=2000)

    # SPCA decompose tt_tt_corr
    # use variable rank
    if l <= 400:
        rank = min(2*l+1, 4)
    else:
        rank = l / 100
    L, S = decompose(tt_tt_corr.real, rank=rank, tol=1.0e-14)

    # if l <= 3:
    #     L, S = decompose(tt_tt_corr.real, rank=l+1, tol=1.0e-14)
    # else:
    #     L, S = decompose(tt_tt_corr.real, rank=10, tol=1.0e-14)
    #     # print matrix_rank(L), matrix_rank(S)
    #     U, s, VT = la.svd(L)
    #     # print np.diff(np.log10(s[:20]))
    #     # print argrelmax(np.diff(np.log10(s[:20])))
    #     rank = argrelmax(np.diff(np.log10(s[:20])))[0][0] + 1
    #     # rank = min(6, rank)
    #     L, S = decompose(tt_tt_corr.real, rank=rank, tol=1.0e-12)

    print matrix_rank(L), matrix_rank(S)

    ls.append(l)
    cls_input.append(cm_cm_corr.real[cv, cv])
    cls.append(S[cv, cv])

ls = np.array(ls)
cls = np.array(cls)
cls_input = np.array(cls_input)
res = cls_input - cls # residual
factor = ls * (ls+1) / (2*np.pi)

# compute input cl
# cl_input = healpy.anafast(cm_map[cv])

# plot cl
plt.figure()
# plt.plot(ls, cl_input, label='input')
plt.plot(ls, cls_input, label='input')
plt.plot(ls, cls, label='recovered')
plt.plot(ls, res, label='residual')
plt.xlabel('$l$')
plt.ylabel('$C_l$')
plt.legend(loc='best')
plt.ylim(0, 1.2e-11)
plt.savefig(out_dir + 'cl_spca.png')
plt.close()

# plot cl
plt.figure()
# plt.plot(ls, factor*cl_input, label='input')
plt.plot(ls, factor*cls_input, label='input')
plt.plot(ls, factor*cls, label='recovered')
plt.plot(ls, factor*res, label='residual')
plt.xlabel('$l$')
plt.ylabel('$l (l + 1) C_l / 2 \pi$')
plt.legend(loc='best')
plt.ylim(0, 2.0e-8)
plt.savefig(out_dir + 'cl_norm_spca.png')
plt.close()
