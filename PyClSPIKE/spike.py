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

def spike(matrixSize, bandwidth, partitionNumber, output, debug = False) :

    # Check the values
    if (partitionNumber < 2) :
        raise ValueError("The partitionNumber has to be at least 2 but it is", partitonNumber)
    elif (matrixSize / partitionNumber < bandwidth) :
        raise ValueError("The number of partitions must be smaller or equal to", matrixSize / bandwidth, "but it is", partitionNumber)
    elif (matrixSize % partitionNumber != 0) :
        raise ValueError("The matrixSize should be devideable by the number of partitions")
    else :
        # Determine the size of each partition
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

    if (output) :
        print config

    # Note: we should put each of the following steps into a separate module/file

    # create Matrices
    A = sparse_creator.create_banded_matrix(matrixSize, bandwidth / 2, bandwidth / 2)
    #x = numpy.ones(matrixSize)
    x = numpy.random.rand(matrixSize)
    b = scipy.sparse.vstack(sparse_creator.create_rhs(A, x))
    #b = scipy.sparse.vstack(numpy.random.rand(matrixSize))

    if (output) :
        print "input A:"
        print A.todense()
        print "input b:"
        print b.todense()

    rhsSize = b.shape[1]
    config['rhsSize'] = rhsSize

    # create prerequirements for OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # compile programm code for CL
    gaussFile = open("gauss.cl", 'r')
    gaussCode = ''.join(gaussFile.readlines())
    program = cl.Program(ctx, gaussCode).build()

    # 1. Pre-processing
    # 1.1 Partitioning of the original system onto different processors
    buffers = partition.partition(config, ctx, A, b, debug)

    if (debug) :
        printMatrix.printMatrix(config, queue, program, buffers[0])

    # 1.2 Factorization of each diagonal block
    # solve A_j[V_j, W_j, G_j] = [(0 ... 0 B_j)T, (C_j 0 ... 0)T, F_j]
    # this step also involves solving of A_j G_j = F_j from (2.1)
    factor.factor(config, ctx, queue, program, buffers)

    if (debug) :
        printMatrix.printMatrix(config, queue, program, buffers[0])

    # At this point we can free the A buffer (buffers[0])
    # TODO make the memory release dependent on some event
    #buffers[0].release()

    # 2. Post-processing
    # 2.1 Solving the reduced system
    buffers = solve.reduced(config, ctx, queue, program, buffers, debug)

    # At this point we can free the SG buffer (buffers[2])
    #buffers[2].release()

    # 2.2 Retrieving the overall solution
    x = solve.final(config, ctx, queue, program, buffers, debug)

    return x

# set basic values
matrixSize = 20000
bandwidth = 4
partitionNumber = 100

#matrixSize = 12
#bandwidth = 2
#partitionNumber = 4

x = spike(matrixSize, bandwidth, partitionNumber, output = True, debug = False)

print "X:"
print x.todense()
