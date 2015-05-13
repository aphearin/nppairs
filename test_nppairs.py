#!/usr/bin/env python

import numpy as np
import nppairs
from time import time

Lbox = 250.0
cell_size = 15.

c = nppairs.NumpyPairCounter(Lbox, cell_size)	

Npts = 1e5
x = np.random.uniform(0, Lbox, Npts)
y = np.random.uniform(0, Lbox, Npts)
z = np.random.uniform(0, Lbox, Npts)
rbins = np.logspace(-1, 1, 10)

start = time()
result = c.num_pairs(x, y, z, rbins)
end = time()
runtime = end-start
print("Total runtime = %.1f seconds" % runtime)