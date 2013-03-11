/**
 * Swap two rows im matrix a
 * @param a the matrix
 * @param c the RHS vector
 * @param rowOne (vertical) position of first row in a
 * @param rowTwo (vertical) position of second row in a
 * @param k (horizontal) position of first non zero
 */
void swap (float* a, float* c, int rowOne, int rowTwo, int k, int n)
{
    if (rowOne == rowTwo) {
        return;
    }

    double tmp;

    //swap matrix
    for (tmp = 0; k < n; k++) {
         tmp = a[rowOne*n+k];
         a[rowOne*n+k] = a[rowTwo*n+k];
         a[rowTwo*n+k] = tmp;
    }

    // swap RHS
    tmp = c[rowOne];
    c[rowOne] = c[rowTwo];
    c[rowTwo] = tmp;
}

/**
 * print matrix of size m×n
 *
 * @param a the matrix
 * @param m count of rows
 * @param n count of columns
 */
__kernel void printMatrix (__global float *a, int m, int n)
{
    printf((__constant char *)"%d x %d\n", m, n);
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            printf((__constant char *)"%f\t", a[i*n+j]);
        }
        printf((__constant char *)"\n");
    }
}

/**
 * Index method, to calculate the index for accessing the matrix
 *
 *
 */
int _(int i, int j, int m, int n, int gid)
{
    return (gid*m+i)*n+j;
}

/**
 * The kernel for solving a linear system with forward elimination and backward substitution
 */
__kernel void gauss(__global float *a, __global float *x, int m, int n)
{
    int i,j,k;
    int gid = get_global_id(0);

    // right hand size
    int rhsize = n-m;
    if (rhsize <= 0) {
        printf((__constant char *)"Partition (%d): The matrix has to be wider than high.\n", gid);
        // write error to error register
        return;
    }

    // forward elimination
    for (k = 0; k < m; k++) {
        if (a[_(k,k,m,n,gid)] == (float)0) {
            printf((__constant char *)"Partition (%d): The matrix is singular at position k=%d (a[k,k]: %f)!\n", gid, k, a[_(k,k,m,n,gid)]);
            // write error to error register
            return;
        }

        // iterate rows
        for (i = k+1; i < m; i++) {
            // iterate columns
            for (j = k+1; j < n; j++) {
                a[_(i,j,m,n,gid)] -= (a[_(i,k,m,n,gid)] / a[_(k,k,m,n,gid)]) * a[_(k,j,m,n,gid)];
            }
            a[_(i,k,m,n,gid)] = 0;
        }
    }

    // backward substitution
    int rhs;
    for (rhs = 0; rhs < rhsize; rhs++) {
        for (i = m-1; i >= 0; i--) {
            x[_(i,rhs,m,rhsize,gid)] = a[_(i,m+rhs,m,n,gid)] / a[_(i,i,m,n,gid)];
            for (j = i+1; j < m; j++) {
                x[_(i,rhs,m,rhsize,gid)] -= (a[_(i,j,m,n,gid)] * x[_(j,rhs,m,rhsize,gid)]) / a[_(i,i,m,n,gid)];
            }
        }
    }
    printf((__constant char *)"Partition (%d) ... done\n", gid);

    return;
}

/**
 * Does the A_j[V_j,W_j,G_j] factorization
 *
 * @param a the global A Matrix
 * @param x the space, where to put the result (V_j,W_j,G_j)
 * @param f the right hand side global F matrix
 * @param n the size of each A_j
 * @param bw the bandwith of the matrix
 */
__kernel void factor(__global float *a, __global float *x,int n, int bw)
{
    int gid = get_global_id(0);
    // extract A_j, B_j and C_j from row
    //a[gid*n+...]
    return;
}
