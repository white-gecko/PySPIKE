# This module partitions the matrix on the processors
# input the matrix, amount of partitions and an OpenCL handler
import numpy
import scipy
import pyopencl as cl

import printMatrix

def partition(config, ctx, A, f, debug = False):

    matrixSize = config['matrixSize']
    bandwidth = config['bandwidth']
    partitionNumber = config['partitionNumber']
    partitionSize = config['partitionSize']
    offdiagonalSize = config['offdiagonalSize']

    Aj = []
    Bj = []
    Cj = []

    for i in range(0, partitionNumber) :
        Aj.append(A[i * partitionSize : (i + 1) * partitionSize, i * partitionSize : (i + 1) * partitionSize])
        if (i+1 < partitionNumber) :
            Bj.append(A[i * partitionSize : (i + 1) * partitionSize, (i + 1) * partitionSize : (i + 1) * partitionSize + offdiagonalSize])
        else :
            Bj.append(scipy.sparse.csr_matrix((partitionSize, offdiagonalSize), dtype=numpy.float))
        if i > 0 :
            Cj.append(A[i * partitionSize : (i + 1) * partitionSize, i * partitionSize - offdiagonalSize : i * partitionSize])
        else :
            Cj.append(scipy.sparse.csr_matrix((partitionSize, offdiagonalSize), dtype=numpy.float))

    if (debug):
        print "A:"
        print A.todense()
        print "Aj:"
        print scipy.sparse.vstack(Aj).todense()
        print "Bj:"
        print scipy.sparse.vstack(Bj).todense()
        print "Cj:"
        print scipy.sparse.vstack(Cj).todense()

    Abcf = numpy.ascontiguousarray(scipy.sparse.hstack([scipy.sparse.vstack(Aj), scipy.sparse.vstack(Bj), scipy.sparse.vstack(Cj), f]).toarray(), dtype=numpy.float32)
    x = numpy.ones((matrixSize, 2 * offdiagonalSize + f.shape[1]), dtype=numpy.float32)

    # move the matrizes to CL-buffer
    mf = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=Abcf)
    x_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=x)

    return [A_buf, x_buf]
