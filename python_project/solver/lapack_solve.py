import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from time import time

class LapackBenchmark(object):
	'''
	Baseline sparse solver from LAPACK for
	sparse matrices (interfaced through scipy)
	'''
	def __init__(self, A, b, x):
		'''
		Initialize with linear banded system, where
		A is of appropriate sparse matrix format (e.g. CSR),
		b is the rhs, and x is the solution vector encoding
		the system Ax = b.
		@param A: scipy sparse matrix (nxn).
		@param b: numpy vector (1xn)
		@param x: numpy vector (1xn)
		'''
		self.A = A
		self.b = b
		self.x = x
	
	def benchmark_spsolve(self, runs = 20, fun = min):
		'''
		Benchmarks the LAPACK sparse matrix solver for the
		given configuration. Runs the solver several times
		and computes a function on the runtime results.
		Prints the results to console (including the avg.
		norm error).
		'''
		bench = []
		x_primes = []
		for i in xrange(runs):
			start 	= time()
			x_prime = spsolve(self.A,self.b)
			stop 	= time()
			bench.append(stop-start)
			x_primes.append(norm(self.x - x_prime))
		res = fun(bench)
		err = sum(x_primes)/len(x_primes)
		print ' '.join(['Runtime over', str(runs), 'runs:', str(res)])
		print ' '.join(['Average error:', str(err)])
