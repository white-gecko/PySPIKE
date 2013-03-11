#include<stdio.h>
#include<stdlib.h>

/**
 * Find the row with the maximum first element
 * @param a the matrix
 * @param k the current position in algorithm
 * @param n the size of the matrix
 */
int argMax (float* a, int k, int n)
{
    int i, max = k;
    for (i = k; i < n; i++) {
        // a[k*n+i] corresponds to a[k][i]
        if (abs(a[i*n+k]) > abs(a[max*n+k])) {
            max = i;
        }
    }

    return max;
}

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
void printMatrix (float* a, int m, int n)
{
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            printf("%f\t", a[i*n+j]);
        }
        printf("\n");
    }
}

int gauss(float* a, float* x, int m, int n)
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

        /* if with pivoting
        i_max = argMax(a, k, n);

        if (a[i_max*n+k] == 0) {
            printf("The matrix is singular!");
            return -1;
        }
        */

        if (a[k*n+k] == 0) {
            printf("The matrix is singular!");
            return -1;
        }

        // if with pivoting
        //swap(a, c, k, i_max, k, n);

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

int main()
{
    //int i,j,k,i_max;
    //int m = 3, n = 5;
    int m = 5, n = 8;

    float *x = (float *)malloc(2 * m * sizeof(float *));
    float *a = (float *)malloc(m * n * sizeof(float *));
    //a[0] = 7;
    //a[1] = 3;
    //a[2] = -5;
    //a[3] = 0;     // c[0]
    //a[4] = -12;     // c[0]
    //a[n] = -1;
    //a[n+1] = -2;
    //a[n+2] = 4;
    //a[n+3] = 0;     // c[1]
    //a[n+4] = 5;     // c[1]
    //a[2*n] = -4;
    //a[2*n+1] = 1;
    //a[2*n+2] = -3;
    //a[2*n+3] = 1;   // c[2]
    //a[2*n+4] = 1;   // c[2]

    a[0] = 0.417022;
    a[1] = 0.198101;
    a[2] = 0.000000;
    a[3] = 0.000000;
    a[4] = 0.000000;
    a[5] = 0.000000; // c[01]
    a[6] = 0.000000; // c[02]
    a[7] = 0.615124; // c[03]
    a[n] = 0.419195;
    a[n+1] = 0.720325;
    a[n+2] = 0.800745;
    a[n+3] = 0.000000;
    a[n+4] = 0.000000;
    a[n+5] = 0.000000; // c[11]
    a[n+6] = 0.000000; // c[12]
    a[n+7] = 1.940264; // c[13]
    a[2*n] = 0.000000;
    a[2*n+1] = 0.685220;
    a[2*n+2] = 0.000114;
    a[2*n+3] = 0.968262;
    a[2*n+4] = 0.000000;
    a[2*n+5] = 0.000000; // c[21]
    a[2*n+6] = 0.000000; // c[22]
    a[2*n+7] = 1.653595; // c[23]
    a[3*n] = 0.000000;
    a[3*n+1] = 0.000000;
    a[3*n+2] = 0.204452;
    a[3*n+3] = 0.302333;
    a[3*n+4] = 0.313424;
    a[3*n+5] = 0.000000;
    a[3*n+6] = 0.000000;
    a[3*n+7] = 0.820209;
    a[4*n] = 0.000000;
    a[4*n+1] = 0.000000;
    a[4*n+2] = 0.000000;
    a[4*n+3] = 0.878117;
    a[4*n+4] = 0.146756;
    a[4*n+5] = 0.692323;
    a[4*n+6] = 0.000000;
    a[4*n+7] = 1.717196;

    printMatrix(a, m, n);
    printf("\n");

    gauss(a, x, m, n);

    printf("a:\n");
    printMatrix(a, m, n);
    printf("\n");

    printf("x:\n");
    printMatrix(x, m, 3);
    printf("\n");

    return 0;
}
