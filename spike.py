#! /usr/bin/env python

# Python implementation of the SPIKE algorithm
# As described in Polizzi et.al. "A parallel hybrid banded system solver: the SPIKE algorithm"
# And Polizzi et.al. "SPIKE: A parallel environment for solving banded linear systems"
#
# This module is the main program to run a complete SPIKE algorithm on a Banded Linear System
#
# (c) 2013 Paul Mayer and Natanael Arndt <arndtn@gmail.com>

from numpy import *
from scipy.sparse import *
from scipy import *
import partition
import factor
import solve

# Determine the number of partitions/processors

# Note: we should put each of the following steps into a separate module/file

# 1. Pre-processing
# 1.1 Partitioning of the original system onto different processors
# 1.2 Factorization of each diagonal block
# 2. Post-processing
# 2.1 Solving the reduced system
# 2.2 Retrieving the overall solution
