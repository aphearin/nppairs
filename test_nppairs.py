#!/usr/bin/env python

import numpy as np
import nppairs
import time

Lbox = 250.0
cell_size = 15.

c = nppairs.NumpyPairCounter(Lbox, cell_size)	

Npts = 1e6
x = np.random.uniform(0, Lbox, Npts)
y = np.random.uniform(0, Lbox, Npts)
z = np.random.uniform(0, Lbox, Npts)

idx_sorted, slice_array = c.cell_structure(x, y, z)
x = x[idx_sorted]
y = y[idx_sorted]
z = z[idx_sorted]


ax, ay, az = c.adjacent_points(x, y, z, slice_array, 0) 



