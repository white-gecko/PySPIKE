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
matrixSize = 20000
bandwidth = 2
partitionNumber = 25
runs = 20
fun = min

args, opts = getopt.getopt(sys.argv[1:], "n:b:p:r:", ["matrixsize=", "bandwidth=", "partitions=", "runs="])

for opt, arg in args:
    if opt in ("-n", "--matrixsize"):
        matrixSize = int(arg)
    elif opt in ("-b", "--bandwidth"):
        bandwidth = int(arg)
    elif opt in ("-p", "--partitions"):
        partitionNumber = int(arg)
    elif opt in ("-r", "--runs"):
        runs = int(arg)

config = {
    'matrixSize': matrixSize,
    'bandwidth': bandwidth,
    'partitionNumber': partitionNumber,
}

# create Matrices
A = sparse_creator.create_banded_matrix(matrixSize, bandwidth / 2, bandwidth / 2)
#x = numpy.ones(matrixSize)
#x = numpy.random.rand(matrixSize)
#b = scipy.sparse.vstack(sparse_creator.create_rhs(A, x))
x_hat = np.ones(matrixSize, dtype=np.float32)
b = sp.sparse.vstack(sparse_creator.create_rhs(A, x_hat))

x_primes = []
bench    = []
for i in xrange(runs):
    x_prime, t = spike.spike(A, b, config, False)
    bench.append(t)
    x_primes.append(norm(x_hat - x_prime))
    print "."
res = fun(bench)
avg = float(sum(bench))/len(bench)
err = sum(x_primes)/len(x_primes)
print ' '.join(['Runtime over', str(runs), 'runs:', str(res), 'avg:', str(avg)])
print ' '.join(['Average error:', str(err)])

# lapack bench
b = sparse_creator.create_rhs(A, x_hat)
lapack_sparse = LapackBenchmark(A,b,x_hat)
lapack_sparse.benchmark_spsolve()
#print "X:"
#print x.todense()
