#! /usr/bin/env python
# The module solves the partitions
import numpy as np
#from numpy import matrix
import scipy as sp
from scipy import linalg
from scipy import sparse
import pyopencl as cl

# solve SX = G
# this step doesn't seam to be parallelizable
def reduced(config, ctx, queue, program, x, debug=False):
    matrixSize = config['matrixSize']
    bandwidth = config['bandwidth']
    partitionNumber = config['partitionNumber']
    partitionSize = config['partitionSize']
    offdiagonalSize = config['offdiagonalSize']
    rhsSize = config['rhsSize']

    Vj = sparse.csr_matrix(x[0 : matrixSize, 0 : offdiagonalSize])
    Wj = sparse.csr_matrix(x[0 : matrixSize, offdiagonalSize : 2 * offdiagonalSize])
    Gj = sparse.csr_matrix(x[0 : matrixSize, 2 * offdiagonalSize :2 * offdiagonalSize + rhsSize])

    if (debug) :
        print Vj
        print Wj
        print Gj

    redV = []
    redW = []
    redG = []

    for i in range(0, partitionNumber) :
        redV.append(topbottom(Vj, i, partitionSize, offdiagonalSize))
        redW.append(topbottom(Wj, i, partitionSize, offdiagonalSize))
        redG.append(topbottom(Gj, i, partitionSize, offdiagonalSize))

    # The shape of the matrix is:
    #
    #   Im  V1  0   0
    #   W2  Im  V2  0
    #   0   W3  Im  V3
    #   0   0   W4  Im

    ones = np.ones(2*offdiagonalSize)
    Im = sp.sparse.dia_matrix((ones, [0]), shape=(2*offdiagonalSize, 2*offdiagonalSize))
    Zero = sp.sparse.coo_matrix((2*offdiagonalSize, offdiagonalSize))

    diag = []
    for i in range(0, partitionNumber) :
        row = []
        if (i == 0) :
            row.append(Im)
            row.append(redV[i])
            row.append(Zero)
            for j in range(2, partitionNumber) :
                row.append(Zero)
                row.append(Zero)
        elif (i+1 < partitionNumber) :
            for j in range(2, i+1) :
                row.append(Zero)
                row.append(Zero)
            row.append(Zero)
            row.append(redW[i])
            row.append(Im)
            row.append(redV[i])
            row.append(Zero)
            for j in range(2, partitionNumber-i) :
                row.append(Zero)
                row.append(Zero)
        else :
            for j in range(2, partitionNumber) :
                row.append(Zero)
                row.append(Zero)
            row.append(Zero)
            row.append(redW[i])
            row.append(Im)
        diag.append(sparse.hstack(row))

    #redS = sparse.block_diag(diag).todense()
    redS = sparse.vstack(diag)
    redG = sparse.vstack(redG)
    SG = np.ascontiguousarray(sparse.hstack([redS, redG]).todense(), dtype=np.float32)

    print "redSG:"
    print SG

    x = np.ones(redG.shape, dtype=np.float32)

    # move the matrizes to CL-buffer
    mf = cl.mem_flags
    SG_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=SG)
    x_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=x)

    kernel = program.gauss
    kernel.set_scalar_arg_dtypes([None, None, np.int32, np.int32])

    kernel(queue, (1,), None, SG_buf, x_buf, np.int32(SG.shape[0]), np.int32(SG.shape[1]))
    cl.enqueue_copy(queue, x, x_buf)

    print x

def topbottom(vector, i, partitionSize, offdiagonalSize):
    return sparse.vstack([
        vector[i * partitionSize : i * partitionSize + offdiagonalSize, 0 : offdiagonalSize], # top
        vector[(i+1) * partitionSize - offdiagonalSize : (i + 1) * partitionSize, 0 : offdiagonalSize]  # bottom
    ])

def final(config, queue, buffers):
    return
