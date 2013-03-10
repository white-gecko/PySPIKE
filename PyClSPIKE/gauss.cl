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

__kernel void gauss(__global float *a, __global float *x, int m, int n)
{
    int i,j,k;

    // right hand size
    int rhsize = n-m;
    if (rhsize <= 0) {
        printf((__constant char *)"The matrix has to be wider than high.");
        // write error to error register
        return;
    }

    // forward elimination
    for (k = 0; k < m; k++) {
        if (a[k*n+k] == 0) {
            printf((__constant char *)"The matrix is singular!");
            // write error to error register
            return;
        }

        // iterate rows
        for (i = k+1; i < m; i++) {
            // iterate columns
            for (j = k+1; j < n; j++) {
                a[i*n+j] = a[i*n+j] - (a[i*n+k] / a[k*n+k]) * a[k*n+j];
            }
            a[i*n+k] = 0;
        }
    }

    // backward substitution
    int rhs;
    for (rhs = 0; rhs < rhsize; rhs++) {
        for (i = m-1; i >= 0; i--) {
            x[i*rhsize+rhs] = a[i*n+m+rhs] / a[i*n+i];
            for (j = i+1; j < n; j++) {
                x[i*rhsize+rhs] -= (a[i*n+j] * x[j*rhsize+rhs]) / a[i*n+i];
            }
        }
    }

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
