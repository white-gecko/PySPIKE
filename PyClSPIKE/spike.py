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

import pyopencl as cl
from time import time

from utils import sparse_creator

import partition
import factor
import solve

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

    config = {
        'matrixSize': matrixSize,
        'bandwidth': bandwidth,
        'partitionNumber': partitionNumber,
        'partitionSize': partitionSize,
        'offdiagonalSize': (bandwidth)/2 #(bandwidth-1)/2
    }

    # Note: we should put each of the following steps into a separate module/file

    # create Matrices
    A = sparse_creator.create_banded_matrix(matrixSize, bandwidth / 2, bandwidth / 2)
    x = numpy.ones(matrixSize)
    b = sparse_creator.create_rhs(A, x)

    # create prerequirements for OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # compile programm code for CL
    gaussFile = open("gauss.cl", 'r')
    gaussCode = "".join(gaussFile.readlines())
    print gaussCode
    #gaussCode = "".join([gaussCode, accessFunctions(config)])
    program = cl.Program(ctx, gaussCode).build()

    # 1. Pre-processing
    # 1.1 Partitioning of the original system onto different processors

    newA = []

    for i in range(0, partitionNumber) :
        newA.append(A[i*partitionSize:(i+1)*partitonSize,i*partitionSize:(i+1)*partitonSize])

    print newA
    print type(A)
    print newA[4:6,4:6]
    return

    # move the matrizes to CL-buffer
    mf = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=A)
    x_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)


    # 1.2 Factorization of each diagonal block

    factor.factor(config, ctx, program,  A_buf, x_buf, b_buf)

    # solve A_j[V_j, W_j, G_j] = [(0 ... 0 B_j)T, (C_j 0 ... 0)T, F_j]
    # this step also involves the solving of A_j G_j = F_j from (2.1)
    solve.gauss(config, ctx, program, A_buf, x_buf, v, w, g)

    # 2. Post-processing
    # 2.1 Solving the reduced system

    # solve SX = G
    # this step doesn't seam to be parallelizable

    # 2.2 Retrieving the overall solution

# This method returns functions which are added to the kernel to easily access values in the matrizes using relativ indizes
def accessFunctions(config) :
    aFunction = """
    int {name}(int i, int j, int gid)
    {{
        return gid*{matrixSize}*{partitionSize}+gid*{partitionSize}+i*{matrixSize}+j;
    }}
    """

    bcFunction = """
    int {name}(int i, int j, int gid)
    {{
        int pos = gid*{matrixSize}*{partitionSize}+gid*{partitionSize}+i*{matrixSize}+j;

        if (pos > {matrixSize}*{matrixSize}) {{
            return -1;
        }}

        return pos;
    }}
    """

    # size of B and C ist m = 
    A = aFunction.format(name="A", matrixSize=config['matrixSize'], partitionSize=config['partitionSize'])
    B = bcFunction.format(name="B", matrixSize=config['matrixSize'], partitionSize=config['partitionSize'])
    C = bcFunction.format(name="C", matrixSize=config['matrixSize'], partitionSize=config['partitionSize'])

    return ''.join([A, B, C])

# set basic values
matrixSize = 20000
matrixSize = 10
bandwidth = 100
bandwidth = 4
partitionNumber = 4
partitionNumber = 2

spike(matrixSize, bandwidth, partitionNumber)
