#import pyopencl as cl
import numpy as np
import scipy as sp
import utils
from scipy import sparse
from scipy import linalg
from time import time

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
		
	def ds_factor(self, p, epsilon):
		'''
		DS factorization of A and computation of
		new rhs.
		@param A: A sparse banded matrix (CSR).
		@param b: RHS of Ax = b, nx1 numpy array.
		@param p: The number of matrix partitions.
		'''
		if p % 2 != 0:
			raise ValueError('p has to be a power of 2')
		
		# compute split sizes
		n 		= self.A.shape[0]
		pdim	= n / p
		
		for pcnt in xrange(p):
			start 	= pcnt * pdim
			stop 	= pcnt * pdim + pdim
			Aj 	= self.A[start:stop,:][:,start:stop]
			Bj = sp.sparse.csr_matrix([0])
			Cj = sp.sparse.csr_matrix([0])
			
			# extract Bj
			if (pcnt != p-1):
				bj_first = start
				Bj = self.A[start:stop,:][:,(stop):(stop+pdim)]
				
				for i in xrange(0,pdim):
					
					if len(Bj.indices[Bj.indptr[i]:Bj.indptr[i+1]]) != 0:
						bj_first = i
						break
					
				bj_colmax = np.max(Bj.indices[Bj.indptr[pdim-1]:Bj.indptr[pdim]])
				Bj = Bj[bj_first:,:][:,0:(bj_colmax+1)]
				
			# extract Cj
			if (pcnt != 0):
				Cj = self.A[start:stop,:][:,(start-pdim):start]
				cj_last = start
				cj_colmax = np.min(Cj.indices[Cj.indptr[0]:Cj.indptr[1]])
				iter_range = range(0, pdim)
				list.reverse(iter_range)
				
				for i in iter_range:
					
					if len(Cj.indices[Cj.indptr[i]:Cj.indptr[i+1]]) != 0:
						cj_last = i
						break
					
				Cj = Cj[0:(cj_last+1),:][:,cj_colmax:]
			
			# debug, only small matrices possible!
			#print 'AJ:', Aj.todense()
			#print 'BJ:', Bj.todense()
			#print 'CJ:', Cj.todense()
			
			# LU decomposition & solve for Wj, Vj
			[Pj, Lj, Uj] = linalg.lu(Aj.todense(), False, False)
			
			# solve for spike V_j
			if (pcnt != p-1):
				y_vects		= []
				nj 			= Aj.shape[0]
				start_at 	= nj - bj_first
				for i in Bj.shape[1]:
					y = []
					for j in Bj.shape[1]:
						y.append()
	
	def recursive_spike(self):
		pass


test = utils.create_banded_matrix(20000,100,100,1)
b = np.zeros(3)
x = np.zeros(3)
start = time()
s = SpykeSolver(sp.sparse.csr_matrix(test), b, x)
s.ds_factor(4, 0)
stop = time()

print stop-start
