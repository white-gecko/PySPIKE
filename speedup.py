#! /usr/bin/env python

import sys
import getopt

from PyClSPIKE import spike
from python_project.utils import sparse_creator
from python_project.solver import LapackBenchmark
import numpy as np
import scipy as sp
from time import time
from numpy.linalg import norm

# set basic values
runs = 20
fun = min

# create Matrices

for bw in [2,16,32,128]:
    for n in [2048,4096,8192]:
        A = sparse_creator.create_banded_matrix(n, bw/2, bw/2)
        x_hat = np.ones(n, dtype=np.float32)
        b = sp.sparse.vstack(sparse_creator.create_rhs(A, x_hat))

        for p in [2,4,8,16,32,64,128,256]:

            if n/p <= bw:
                continue

            config = {
                'matrixSize': n,
                'bandwidth': bw,
                'partitionNumber': p,
            }

            bench    = []
            for i in xrange(runs):
                x_prime, t = spike.spike(A, b, config, False)
                bench.append(t)
                sys.stdout.write('.')
                sys.stdout.flush()
            res = fun(bench)
            avg = float(sum(bench))/len(bench)
            print ''
            print 'b=', str(bw), 'n=', str(n), 'p=', str(p) , 'Runtime over', str(runs), 'runs:', str(res), 'avg:', str(avg), 'delta:', str(avg-res)

