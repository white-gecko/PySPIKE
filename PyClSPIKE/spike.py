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
import scipy

import pyopencl as cl
from time import time

from utils import sparse_creator

import partition
import factor
import solve
import printMatrix

def spike(matrixSize, bandwidth, partitionNumber) :

    # Determine the size of each partition
    if (matrixSize / partitionNumber < bandwidth) :
        print "The number of partitions must be smaller or equal to", matrixSize / bandwidth, "but it is", partitionNumber
        return
    elif (matrixSize % partitionNumber != 0) :
        print "The matrixSize should be devideable by the number of partitions"
        return
    else :
        partitionSize = matrixSize / partitionNumber

    # make bandwidth even
    if (bandwidth % 2 != 0) :
        bandwidth += 1

    offdiagonalSize = (bandwidth)/2 #(bandwidth-1)/2
    config = {
        'matrixSize': matrixSize,
        'bandwidth': bandwidth,
        'partitionNumber': partitionNumber,
        'partitionSize': partitionSize,
        'offdiagonalSize': offdiagonalSize
    }

    # Note: we should put each of the following steps into a separate module/file

    # create Matrices
    A = sparse_creator.create_banded_matrix(matrixSize, bandwidth / 2, bandwidth / 2)
    x = numpy.ones(matrixSize)
    b = scipy.sparse.vstack(sparse_creator.create_rhs(A, x))

    config['rhsSize'] = b.shape[1]

    # create prerequirements for OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # compile programm code for CL
    gaussFile = open("gauss.cl", 'r')
    gaussCode = ''.join(gaussFile.readlines())
    program = cl.Program(ctx, gaussCode).build()

    # 1. Pre-processing
    # 1.1 Partitioning of the original system onto different processors
    buffers = partition.partition(config, ctx, queue, program, A, b, debug = False)

    printMatrix.printMatrix(config, queue, program, buffers[0])

    # 1.2 Factorization of each diagonal block
    # solve A_j[V_j, W_j, G_j] = [(0 ... 0 B_j)T, (C_j 0 ... 0)T, F_j]
    # this step also involves solving of A_j G_j = F_j from (2.1)
    factor.factor(config, ctx, queue, program, buffers)

    # 2. Post-processing
    # 2.1 Solving the reduced system
    #solve.gauss(config, ctx, queue, program, buffers)

    # solve SX = G
    # this step doesn't seam to be parallelizable

    # 2.2 Retrieving the overall solution

# set basic values
matrixSize = 20000
matrixSize = 20
bandwidth = 100
bandwidth = 4
partitionNumber = 4
partitionNumber = 4

spike(matrixSize, bandwidth, partitionNumber)
