#! /usr/bin/env python
# This module factorices the partitions
import solve

def factor(config, ctx, program, A_buf, x_buf, f_buf):
    kernel = program.factor
    kernel.set_scalar_arg_dtypes([None, numpy.int32, numpy.int32])

    solve.gauss(ctx, a_buf, x_buf, v, w, g)
    return
