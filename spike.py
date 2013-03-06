#! /usr/bin/env python

# Python implementation of the SPIKE algorithm
# As described in Polizzi et.al. "A parallel hybrid banded system solver: the SPIKE algorithm"
# And Polizzi et.al. "SPIKE: A parallel environment for solving banded linear systems"
#
# This module is the main program to run a complete SPIKE algorithm on a Banded Linear System
#
# (c) 2013 Paul Mayer and Natanael Arndt <arndtn@gmail.com>

from numpy import *
from scipy.sparse.linalg import *
from scipy.sparse import *
from scipy import *
import partition
import factor
import solve

# Determine the number of partitions/processors

# Note: we should put each of the following steps into a separate module/file

# load matrix from .mat file with scipy matlab

# 1. Pre-processing
# 1.1 Partitioning of the original system onto different processors

# define the A_j, B_j and C_j blocks

# 1.2 Factorization of each diagonal block

gaussFile = open("gauss.cl", 'r')
gaussKernelCode = "".join(gaussFile.readlines())

# solve A_j[V_j, W_j, G_j] = [(0 ... 0 B_j)T, (C_j 0 ... 0)T, F_j]
# this step also involves the solving of A_j G_j = F_j from (2.1)

# 2. Post-processing
# 2.1 Solving the reduced system

# solve SX = G
# this step doesn't seam to be parallelizable

# 2.2 Retrieving the overall solution
