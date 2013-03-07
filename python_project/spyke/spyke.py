import pyopencl as cl
import numpy as np
import scipy as sp
from utils import sparse_creator

class SpykeSolver(object):
	'''
	Expects matrix in sparse format (CSR).
	Initializes elements to solve Ax = b.
	x and b are assument column vectors
	'''
	def __init__(self, matrix, rhs): 
		pass

	def recursive_spike(self):
		pass

