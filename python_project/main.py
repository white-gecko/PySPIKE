from utils import sparse_creator
from solver import LapackBenchmark
#from spyke import Spyke
import numpy as np

A = sparse_creator.create_banded_matrix(100000,20,20)
x = np.ones(100000)
b = sparse_creator.create_rhs(A, x)

lapack_sparse = LapackBenchmark(A,b,x)
lapack_sparse.benchmark_spsolve()
