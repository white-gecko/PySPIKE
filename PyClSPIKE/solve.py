#! /usr/bin/env python
# The module solves the partitions
import numpy
import pyopencl as cl

# solve SX = G
# this step doesn't seam to be parallelizable
def reduced(config, queue, buffers, debug=False):
    matrixSize = config['matrixSize']
    bandwidth = config['bandwidth']
    partitionNumber = config['partitionNumber']
    partitionSize = config['partitionSize']
    offdiagonalSize = config['offdiagonalSize']
    rhsSize = config['rhsSize']

    x = numpy.zeros((matrixSize, 2 * offdiagonalSize + rhsSize), dtype=numpy.float32)
    cl.enqueue_copy(queue, x, buffers[1])

    Vj = x[0 : matrixSize, 0 : offdiagonalSize]
    Wj = x[0 : matrixSize, offdiagonalSize : 2 * offdiagonalSize]
    Gj = x[0 : matrixSize, 2 * offdiagonalSize :2 * offdiagonalSize + rhsSize]

    if (debug) :
        print Vj
        print Wj
        print Gj

    redV = []
    redW = []
    redG = []

    for i in range(0, partitionNumber) :
        redV.append([
            Vj[i * partitionSize : i * partitionSize + offdiagonalSize, 0 : offdiagonalSize], # top
            Vj[(i+1) * partitionSize - offdiagonalSize : (i + 1) * partitionSize, 0 : offdiagonalSize]  # bottom
        ])
        redW.append([
            Wj[i * partitionSize : i * partitionSize + offdiagonalSize, 0 : offdiagonalSize], # top
            Wj[(i+1) * partitionSize - offdiagonalSize : (i + 1) * partitionSize, 0 : offdiagonalSize]  # bottom
        ])
        redG.append([
            Gj[i * partitionSize : i * partitionSize + offdiagonalSize, 0 : offdiagonalSize], # top
            Gj[(i+1) * partitionSize - offdiagonalSize : (i + 1) * partitionSize, 0 : offdiagonalSize]  # bottom
        ])

    print redV

def final(config, queue, buffers):
    return
