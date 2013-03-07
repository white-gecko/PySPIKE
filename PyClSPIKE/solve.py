#! /usr/bin/env python
# The module solves the partitions

# execute a cl-kernel

def gauss(ctx, a_buf, x_buf, shape, m, n):
    gaussFile = open("gauss.cl", 'r')
    gaussKernelCode = "".join(gaussFile.readlines())
    prg = cl.Program(ctx, gaussKernelCode).build()
    kernel = prg.gauss
    kernel.set_scalar_arg_dtypes([None, numpy.int32, numpy.int32])
    # I hope it also takes the last column
    # how can I put only k jobs into the queue
    kernel(queue, shape, None, a_buf, x_buf, numpy.int32(m), numpy.int32(n))
