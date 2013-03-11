import pyopencl as cl
import numpy as np
import scipy as sp

class SpykeSolver(object):
	'''
	Expects matrix in sparse format (CSR).
	Initializes elements to solve Ax = b.
	x and b are assument column vectors
	'''
	def __init__(self, A, b, x): 
		self.A = A
		self.b = b
		self.x = x

	def __ds_factor(self, A, b, p):
		'''
		DS factorization of A and computation of
		new rhs.
		@param A: A sparse banded matrix (CSR).
		@param b: RHS of Ax = b, nx1 numpy array.
		@param p: The number of matrix partitions.
		'''
		if not p % 2:
			raise ValueError('p has to be a power of 2')
 
	
	def recursive_spike(self):
		pass

