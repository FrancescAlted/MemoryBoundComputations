# Benchmark to compare the times for evaluating queries.
# Numexpr is needed in order to execute this.

from __future__ import print_function

import math
from time import time

import numpy as np
import numexpr as ne

import bcolz


N = int(1e7)  # the number of elements in the tables
clevel = 5  # the compression level
cname = "blosclz"  # the compressor name, blosclz is usually the fastest in Blosc
#cname = "lz4"     # LZ4 is a very fast compressor too
#cname = "lz4hc"   # LZ4HC compresses more than LZ4, but it compress slower
#cname = "zlib"    # you may want to try this compressor classic too
#cname = "zstd"    # Zstd is a well balanced compressor
sexpr = "(2*x*x + .3*y*y + z + 1) < 100"  # the query to compute

# Uncomment the next for disabling multi-threading
#ne.set_num_threads(1)
#bcolz.set_nthreads(1)

print("Creating tables with 10^%d points..." % int(math.log10(N)))

x = np.arange(N)
y = np.linspace(1, 10, N)
z = np.arange(N) * 10

# Build a ctable making use of above arrays as columns
cparams = bcolz.cparams(clevel=clevel, cname=cname, shuffle=1)
t = bcolz.ctable((x, y, z, x * 2, y + .5, z // 10),
                 names=['x', 'y', 'z', 'xp', 'yp', 'zp'],
                 cparams=cparams)
# The NumPy structured array version
nt = t[:]

del x, y, z  # we are not going to need these arrays anymore

print("Querying '%s' with different containers..." % sexpr)

t0 = time()
out = [r for r in nt[eval(sexpr, {'x': nt['x'], 'y': nt['y'], 'z': nt['z']})]]
print("Time for numpy (vm=python)-->  *** %.3fs ***" % (time() - t0,))

t0 = time()
out = [r for r in nt[ne.evaluate(sexpr, {'x': nt['x'], 'y': nt['y'], 'z': nt['z']})]]
print("Time for numpy (vm=numexpr)-->  *** %.3fs ***" % (time() - t0,))

print("Compression ratio for ctable container: %.2fx" % (t.nbytes / float(t.cbytes)))

t0 = time()
cout = [r for r in t.where(sexpr, vm='python')]
print("Time for bcolz (vm=python)--> *** %.3fs ***" % (time() - t0,))

t0 = time()
cout = [r for r in t.where(sexpr, vm='numexpr')]
print("Time for bcolz (vm=numexpr)--> *** %.3fs ***" % (time() - t0,))

t0 = time()
cout = [r for r in t.where(sexpr, vm='dask')]
print("Time for bcolz (vm=dask) --> *** %.3fs ***" % (time() - t0,))
