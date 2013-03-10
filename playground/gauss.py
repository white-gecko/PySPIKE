#! /usr/bin/env python

import pyopencl as cl
import numpy
from time import time
#import numpy.linalg as la

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 4

a = numpy.empty(n*(n-1), dtype=numpy.float32)

a[0] = 7
a[1] = 3
a[2] = -5
a[3] = -12 # c[0]
a[n] = -1
a[n+1] = -2
a[n+2] = 4
a[n+3] = 5 # c[1]
a[2*n] = -4
a[2*n+1] = 1
a[2*n+2] = -3
a[2*n+3] = 1 # c[2]

print a

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

prg = cl.Program(ctx, """
    __kernel void eliminate(__global float *a, int k, int n)
    {
        int gid = get_global_id(0);
        int i;
        int ak = k*n+k;
        int ai = k*n+gid;
        for (i = k+1; i < n; i++) {
            a[i*n+gid] = a[i*n+gid] - (a[i*n+k] / a[ak]) * a[ai];
            //a[i*n+gid] = gid;
        }
    }
""").build()

for k in range(n-1):
    kernel = prg.eliminate
    kernel.set_scalar_arg_dtypes([None, numpy.int32, numpy.int32])
    # I hope it also takes the last column
    # how can I put only k jobs into the queue
    event = kernel(queue, (n-k,), None, a_buf, numpy.int32(k), numpy.int32(n))
    #for (i = k+1; i < n; i++) {
    #    a[i*n+k] = 0
    #}
    # We need to wait for all jobs on each loop
    t1 = time()
    cl.enqueue_barrier(queue)
    t2 = time()
    cl.enqueue_copy(queue, a, a_buf)
    t3 = time()
    #print("t1: ", t1-t2, " t2: ", t2-t3)
    print("k: ", k)
    print(a)

cl.enqueue_copy(queue, a, a_buf)

#print a
