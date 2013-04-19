import numpy as np
from scipy.sparse import diags

def create_banded_matrix(n, k1, k2, seed = 1):
	'''
	Creates a nxn banded matrix with given bandwidth
	and of given data type.
	@param n: matrix dimensionality.
	@param k1: left half-bandwidth.
	@param k2: right half-bandwidth.
	'''
	if k1 < 0 or k2 < 0:
		raise ValueError("half-bandwidths have to be >= 0")
	diagonals = []
	np.random.seed(1)
	for i in xrange(k1+1):
		diagonals.append(np.random.rand(n-i))
	for i in xrange(1,k2+1):
		diagonals.append(np.random.rand(n-i))
	return diags(diagonals, [-x for x in range(k1+1)] + range(1,k2+1), format="csr", dtype=np.float32)
	
	
def create_rhs(matrix, result_vect):
	'''
	Creates the RHS of a linear system Ax = b, where
	A is a banded nxn matrix of sparse matrix format.
	@param matrix: sparse representation of the banded
	matrix A (nxn).
	@param result_vect: corresponds to x (1xn).
	@return b, the RHS of the linear system which should
	result in solve(A,b) = rhs (1xn vector).
	'''
	if len(result_vect) != matrix.shape[0]:
		raise ValueError("Sizes do not match")
	return matrix * result_vect
