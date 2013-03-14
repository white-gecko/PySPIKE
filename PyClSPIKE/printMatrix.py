import numpy

def printMatrix(config, queue, program, A_buf, m = 0, n = 0):
    kernel = program.printMatrix
    kernel.set_scalar_arg_dtypes([None, numpy.int32, numpy.int32])

    if (m == 0) :
        m = numpy.int32(config['matrixSize'])
    if (n == 0) :
        n = numpy.int32(config['partitionSize'] + 2 * config['offdiagonalSize'] + config['rhsSize'])

    kernel(queue, (1,), None, A_buf, m, n)
