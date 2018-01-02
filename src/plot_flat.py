import os
import numpy as np
import h5py
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


conv_beam = config.conv_beam
D = config.D

if conv_beam:
    out_dir = '../results/flat_sky/conv_%.1f/' % D
else:
    out_dir = '../results/flat_sky/no_conv/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


with h5py.File(out_dir+'ps.hdf5', 'r') as f:
    nf = f['map'].shape[0]
    cf = nf/2 # central frequency point to plot
    lonra = f['map'].attrs['lonra'].tolist()
    latra = f['map'].attrs['latra'].tolist()
    ps = f['map'][cf]
with h5py.File(out_dir+'ga.hdf5', 'r') as f:
    ga = f['map'][cf]
with h5py.File(out_dir+'cm.hdf5', 'r') as f:
    cm = f['map'][cf]


# plot ps
plt.figure(figsize=(13, 5))
plt.imshow(ps, origin='lower', aspect='equal', extent=lonra+latra, vmin=0, vmax=10)
plt.colorbar()
plt.xlabel(r'$\alpha$ / degree')
plt.ylabel(r'$\delta$ / degree')
# plt.colorbar(orientation='horizontal')
plt.savefig(out_dir+'ps_%d.png' % cf)
plt.close()

# plot ga
plt.figure(figsize=(13, 5))
plt.imshow(ga, origin='lower', aspect='equal', extent=lonra+latra, vmin=0, vmax=10)
plt.colorbar()
plt.xlabel(r'$\alpha$ / degree')
plt.ylabel(r'$\delta$ / degree')
plt.savefig(out_dir+'ga_%d.png' % cf)
plt.close()

# plot cm
plt.figure(figsize=(13, 5))
plt.imshow(cm, origin='lower', aspect='equal', extent=lonra+latra)
plt.colorbar()
plt.xlabel(r'$\alpha$ / degree')
plt.ylabel(r'$\delta$ / degree')
plt.savefig(out_dir+'cm_%d.png' % cf)
plt.close()

# plot tt
plt.figure(figsize=(13, 5))
plt.imshow(ps+ga+cm, origin='lower', aspect='equal', extent=lonra+latra, vmin=0, vmax=10)
plt.colorbar()
plt.xlabel(r'$\alpha$ / degree')
plt.ylabel(r'$\delta$ / degree')
plt.savefig(out_dir+'tt_%d.png' % cf)
plt.close()