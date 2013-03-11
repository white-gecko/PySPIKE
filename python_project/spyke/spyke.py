#import pycuda.driver as cuda
#import pycuda.autoinit
#from pycuda.compiler import SourceModuleimport
import numpy as np
import scipy as sp
import utils
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
			cj_last = -1
			bj_first = -1
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
			
			# solve for spikes V_j, W_j and G_j
			nj 			= Aj.shape[0]
			b_width		= Bj.shape[1]
			c_width		= Cj.shape[1]
			y_vj_vects	= np.zeros(shape=(Bj.shape[0], Bj.shape[1]), dtype=np.float32)
			y_wj_vects	= np.zeros(shape=(Cj.shape[0], Cj.shape[1]), dtype=np.float32)
			y_gj_vect	= np.zeros(shape=(nj), dtype=np.float32)
			
			if (pcnt == 0):
				wj_stopat = -1
			else:
				wj_stopat = cj_last
				
			print wj_stopat
				
			if (pcnt == p-1):
				vj_startat = stop + 1
			else:
				vj_startat = bj_first
				
			vj_startat = nj - bj_first
			print vj_startat, wj_stopat
			
			for i in xrange(nj):
				lrange = Lj[i,0:i].transpose()
				if i < wj_stopat - 1:

					for j in xrange(b_width):
						print "lengths", len(Lj[i,0:i]),len(y_vj_vects[j,0:i])
						y_vj_vects[j, i] = (1./Lj[i,i] - (np.dot(lrange,y_vj_vects[j, 0:i])))
				
				if i > vj_startat + 1:
					for j in xrange(c_width):
						print "lengths", len(Lj[i,0:i]),len(y_wj_vects[j,0:i-vj_startat])
						y_wj_vects[j, i] = (1./Lj[i,i] * (np.sum(Lj[i,vj_startat:i] * y_wj_vects[j,0:i-vj_startat])))
					
				y_gj_vect[i] = (1./Lj[i,i] - (np.dot(lrange,y_gj_vect[0:i])))
				#print y_gj_vect[0:i]


	def recursive_spike(self):
		pass


test = utils.create_banded_matrix(100,6,6,1)
b = np.zeros(3)
x = np.zeros(3)
start = time()
s = SpykeSolver(sp.sparse.csr_matrix(test), b, x)
s.ds_factor(4, 0)
stop = time()

print stop-start
