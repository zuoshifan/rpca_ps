import os
import numpy as np
import h5py
from cora.util import hputil
import config


conv_beam = config.conv_beam
D = config.D
nside = config.nside

if conv_beam:
    out_dir = '../results/alm/conv_%.1f/' % D
else:
    out_dir = '../results/alm/no_conv/'
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


ps_alm = hputil.sphtrans_sky(ps_map)
ga_alm = hputil.sphtrans_sky(ga_map)
cm_alm = hputil.sphtrans_sky(cm_map)

ps_alm_name = 'alm_pointsource_%d_700_800_256.hdf5' % nside
ga_alm_name = 'alm_galaxy_%d_700_800_256.hdf5' % nside
cm_alm_name = 'alm_21cm_%d_700_800_256.hdf5' % nside

# save alms
with h5py.File(out_dir+ps_alm_name, 'w') as f:
    ps = f.create_dataset('alm', data=ps_alm)
    ps.attrs['axes'] = '(freq, l, m)'
with h5py.File(out_dir+ga_alm_name, 'w') as f:
    ga = f.create_dataset('alm', data=ga_alm)
    ga.attrs['axes'] = '(freq, l, m)'
with h5py.File(out_dir+cm_alm_name, 'w') as f:
    cm = f.create_dataset('alm', data=cm_alm)
    cm.attrs['axes'] = '(freq, l, m)'
