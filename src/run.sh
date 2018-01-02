#!/usr/bin/env bash

# generate beam convolved sky maps
# python convolve_beam.py

# crop to get a flat sky area
# python crop_map.py

# plot flat skys
# python plot_flat.py

# Fourier transform the flat sky areas
python check_flat.py
python fourier_flat.py

# compute cl for full sky maps
python cl.py

echo done