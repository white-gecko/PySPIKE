#! /usr/bin/env python
# This module factorices the partitions
import numpy

def factor(config, ctx, queue, program, buffers):
    kernel = program.gauss
    kernel.set_scalar_arg_dtypes([None, None, numpy.int32, numpy.int32])

    kernel(
        queue,
        (config['partitionNumber'],),
        None,
        buffers[0],  # A matrix
        buffers[1],  # x matrix
        numpy.int32(config['partitionSize']),
        numpy.int32(config['partitionSize'] + 2 * config['offdiagonalSize'] + config['rhsSize'])
    )
    return
