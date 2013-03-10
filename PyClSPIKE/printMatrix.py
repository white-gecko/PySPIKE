import numpy

def printMatrix(config, queue, program, A_buf):
    kernel = program.printMatrix
    kernel.set_scalar_arg_dtypes([None, numpy.int32, numpy.int32])

    # I hope it also takes the last column
    # how can I put only k jobs into the queue
    kernel(
        queue,
        (1,),
        None,
        A_buf,  # A matrix
        numpy.int32(config['matrixSize']),
        numpy.int32(config['partitionSize'] + 2 * config['offdiagonalSize'] + config['rhsSize'])
    )
