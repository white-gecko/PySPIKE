#! /usr/bin/env python

# Python implementation of the SPIKE algorithm
# As described in Polizzi et.al. "A parallel hybrid banded system solver: the SPIKE algorithm"
# And Polizzi et.al. "SPIKE: A parallel environment for solving banded linear systems"
#
# This module is the main program to run a complete SPIKE algorithm on a Banded Linear System
#
# (c) 2013 Paul Mayer and Natanael Arndt <arndtn@gmail.com>

import sys
import numpy
from scipy.sparse.linalg import *
from scipy.sparse import *
from scipy import *
import scipy.io as sio

import pyopencl as cl
from time import time

import partition
import factor
import solve

# Determine the number of partitions/processors

# Note: we should put each of the following steps into a separate module/file

# load matrix from .mat file with scipy matlab
matrixFileName = sys.argv[1]

print "loading matrix file:", matrixFileName
matlab = sio.loadmat(matrixFileName)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
# 1. Pre-processing
# 1.1 Partitioning of the original system onto different processors

# define the A_j, B_j and C_j blocks

a = numpy.random.rand(50000).astype(numpy.float32)
x = numpy.random.rand(50000).astype(numpy.float32)
f = numpy.random.rand(50000).astype(numpy.float32)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
x_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
f_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)

# 1.2 Factorization of each diagonal block

factor.factor(ctx, a_buf, x_buf, f_buf)

# solve A_j[V_j, W_j, G_j] = [(0 ... 0 B_j)T, (C_j 0 ... 0)T, F_j]
# this step also involves the solving of A_j G_j = F_j from (2.1)
solve.gauss(ctx, a_buf, x_buf, v, w, g)

# 2. Post-processing
# 2.1 Solving the reduced system

# solve SX = G
# this step doesn't seam to be parallelizable

# 2.2 Retrieving the overall solution
