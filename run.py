#! /usr/bin/env python

from PyClSPIKE import spike
from python_project.utils import sparse_creator
import numpy
import scipy

# set basic values
matrixSize = 20000
bandwidth = 6
partitionNumber = 100

config = {
    'matrixSize': matrixSize,
    'bandwidth': bandwidth,
    'partitionNumber': partitionNumber,
}

# Note: we should put each of the following steps into a separate module/file

# create Matrices
A = sparse_creator.create_banded_matrix(matrixSize, bandwidth / 2, bandwidth / 2)
#x = numpy.ones(matrixSize)
#x = numpy.random.rand(matrixSize)
#b = scipy.sparse.vstack(sparse_creator.create_rhs(A, x))
b = scipy.sparse.csr_matrix(numpy.random.rand(matrixSize, 2))

x = spike.spike(A, b, config)

print "X:"
print x.todense()
