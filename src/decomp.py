import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy import linalg as la
import h5py
from rpca import ialm
import healpy
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


conv_beam = config.conv_beam
D = config.D

if conv_beam:
    out_dir = '../results/decomp/conv_%.1f/' % D
else:
    out_dir = '../results/decomp/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if conv_beam:
    in_dir = '../results/corr_data/full_sky/conv_%.1f/' % D
else:
    out_dir = '../results/corr_data/full_sky/no_conv/'
with h5py.File(in_dir+'corr.hdf5', 'r') as f:
    cm_cm_corr = f['cm_cm'][:]
    tt_tt_corr = f['tt_tt'][:]

# L, S = ialm(tt_tt_corr, tol1=1.0e-6, verbose=True)
L, S = ialm(tt_tt_corr, verbose=True)

print matrix_rank(L)
print matrix_rank(S)
print la.norm(cm_cm_corr - S, ord='fro') / la.norm(cm_cm_corr, ord='fro')
print np.allclose(L, L.T), np.allclose(S, S.T)

# save data to file
with h5py.File(out_dir + 'decomp.hdf5', 'w') as f:
    f.create_dataset('tt_tt', data=tt_tt_corr)
    f.create_dataset('cm_cm', data=cm_cm_corr)
    f.create_dataset('L', data=L)
    f.create_dataset('S', data=S)

M = tt_tt_corr
R = cm_cm_corr

# plot
plt.figure()
plt.subplot(221)
plt.imshow(M, origin='lower', aspect='equal')
plt.colorbar()
plt.subplot(222)
# plt.imshow(M-L-S, origin='lower', aspect='equal')
plt.imshow(R-S, origin='lower', aspect='equal')
plt.colorbar()
plt.subplot(223)
plt.imshow(L, origin='lower', aspect='equal')
plt.colorbar()
plt.subplot(224)
plt.imshow(S, origin='lower', aspect='equal')
plt.colorbar()
plt.savefig('decomp.png')
plt.close()
