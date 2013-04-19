#! /usr/bin/env python

# Python implementation of the SPIKE algorithm
# As described in Polizzi et.al. "A parallel hybrid banded system solver: the SPIKE algorithm"
# And Polizzi et.al. "SPIKE: A parallel environment for solving banded linear systems"
#
# This module is the main program to run a complete SPIKE algorithm on a Banded Linear System
#
# (c) 2013 Paul Mayer and Natanael Arndt <arndtn@gmail.com>

import sys
import os
import numpy
import scipy

import pyopencl as cl
from time import time

import partition
import factor
import solve

def spike(A, b, config, output = True, debug = False) :

    # Check the values
    if (config['partitionNumber'] < 2) :
        raise ValueError("The partitionNumber has to be at least 2 but it is", config['partitionNumber'])
    elif (config['matrixSize'] / config['partitionNumber'] < config['bandwidth']) :
        raise ValueError("The number of partitions must be smaller or equal to", config['matrixSize'] / config['bandwidth'], "but it is", config['partitionNumber'])
    elif (config['matrixSize'] % config['partitionNumber'] != 0) :
        raise ValueError("The matrixSize should be devideable by the number of partitions")
    else :
        # Determine the size of each partition
        config['partitionSize'] = config['matrixSize'] / config['partitionNumber']

    # make bandwidth even
    if (config['bandwidth'] % 2 != 0) :
        config['bandwidth'] += 1

    config['rhsSize'] = b.shape[1]

    config['offdiagonalSize'] = config['bandwidth']/2 #(bandwidth-1)/2

    if (output) :
        print config
        print "input A:"
        print A.todense()
        print "input b:"
        print b.todense()

    # create prerequirements for OpenCL
    ctx = cl.create_some_context(True)
    #for platform in cl.get_platforms() :
    #    devices = platform.get_devices(cl.device_type.CPU)
    #    ctx = cl.Context(devices)
    queue = cl.CommandQueue(ctx)

    # compile programm code for CL
    directory = os.path.dirname(os.path.realpath(__file__))
    gaussFile = open(os.path.join(directory, "gauss.cl"), 'r')
    gaussCode = ''.join(gaussFile.readlines())
    program = cl.Program(ctx, gaussCode).build()

    # 1. Pre-processing
    # 1.1 Partitioning of the original system onto different processors
    start = time()
    buffers = partition.partition(config, ctx, A, b, debug)

    # 1.2 Factorization of each diagonal block
    # solve A_j[V_j, W_j, G_j] = [(0 ... 0 B_j)T, (C_j 0 ... 0)T, F_j]
    # this step also involves solving of A_j G_j = F_j from (2.1)
    factor.factor(config, ctx, queue, program, buffers)

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
    stop = time()
    rt = stop-start
    return [x, rt]
