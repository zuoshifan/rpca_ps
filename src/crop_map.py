import os
import numpy as np
import h5py
import healpy as hp
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


conv_beam = config.conv_beam
D = config.D
nside = config.nside

lonra=[-15, 15] # degree
latra=[-10, 10] # degree

if conv_beam:
    out_dir = '../results/flat_sky/conv_%.1f/' % D
else:
    out_dir = '../results/flat_sky/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if conv_beam:
    map_dir = '../results/conv_beam/conv_%.1f/' % D
    ps_name = map_dir + 'smooth_pointsource_%d_700_800_256.hdf5' % nside
    ga_name = map_dir + 'smooth_galaxy_%d_700_800_256.hdf5' % nside
    cm_name = map_dir + 'smooth_21cm_%d_700_800_256.hdf5' % nside
    with h5py.File(ps_name, 'r') as f:
        ps_map = f['map'][:]
    with h5py.File(ga_name, 'r') as f:
        ga_map = f['map'][:]
    with h5py.File(cm_name, 'r') as f:
        cm_map = f['map'][:]
else:
    map_dir = '../sky_map/'
    ps_name = map_dir + 'sim_pointsource_%d_700_800_256.hdf5' % nside
    ga_name = map_dir + 'sim_galaxy_%d_700_800_256.hdf5' % nside
    cm_name = map_dir + 'sim_21cm_%d_700_800_256.hdf5' % nside
    with h5py.File(ps_name, 'r') as f:
        ps_map = f['map'][:, 0, :]
    with h5py.File(ga_name, 'r') as f:
        ga_map = f['map'][:, 0, :]
    with h5py.File(cm_name, 'r') as f:
        cm_map = f['map'][:, 0, :]

nfreq = ps_map.shape[0]
xsize, ysize = 120, 80
ps_flat = np.zeros((nfreq, ysize, xsize), dtype=ps_map.dtype)
ga_flat = np.zeros((nfreq, ysize, xsize), dtype=ps_map.dtype)
cm_flat = np.zeros((nfreq, ysize, xsize), dtype=ps_map.dtype)

cf = nfreq / 2
for fi in range(nfreq):
    ps_flat[fi] = hp.cartview(ps_map[fi], xsize=xsize, ysize=ysize, lonra=lonra, latra=latra, return_projected_map=True)
    ga_flat[fi] = hp.cartview(ga_map[fi], xsize=xsize, ysize=ysize, lonra=lonra, latra=latra, return_projected_map=True)
    cm_flat[fi] = hp.cartview(cm_map[fi], xsize=xsize, ysize=ysize, lonra=lonra, latra=latra, return_projected_map=True)


# save to files
with h5py.File(out_dir+'ps.hdf5', 'w') as f:
    f.create_dataset('map', data=ps_flat)
    f['map'].attrs['lonra'] = lonra
    f['map'].attrs['latra'] = latra
with h5py.File(out_dir+'ga.hdf5', 'w') as f:
    f.create_dataset('map', data=ga_flat)
    f['map'].attrs['lonra'] = lonra
    f['map'].attrs['latra'] = latra
with h5py.File(out_dir+'cm.hdf5', 'w') as f:
    f.create_dataset('map', data=cm_flat)
    f['map'].attrs['lonra'] = lonra
    f['map'].attrs['latra'] = latra