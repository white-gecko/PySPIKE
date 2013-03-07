#include<stdio.h>
#include<stdlib.h>

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

__kernel int gauss(__global float *a, __global float *x, __global int m, __global int n)
{
    int i,j,k;

    // right hand size
    int rhsize = n-m;
    if (rhsize <= 0) {
        printf("The matrix has to be wider than high.");
        return -1;
    }

    // forward elimination
    for (k = 0; k < m; k++) {
        if (a[k*n+k] == 0) {
            printf("The matrix is singular!");
            return -1;
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

    return 0;
}