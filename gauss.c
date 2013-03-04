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

int main()
{
    int i,j,k,n = 3, i_max;

    float *x = (float *)malloc(n * sizeof(float *));
    float *c = (float *)malloc(n * sizeof(float *));
    float *a = (float *)malloc(n * n * sizeof(float *));
    a[0] = 7;
    a[1] = 3;
    a[2] = -5;
    a[n] = -1;
    a[n+1] = -2;
    a[n+2] = 4;
    a[2*n] = -4;
    a[2*n+1] = 1;
    a[2*n+2] = -3;
    c[0] = -12;
    c[1] = 5;
    c[2] = 1;

    printMatrix(a, n, n);
    printf("\n");

    for (k = 0; k < n; k++) {
        i_max = argMax(a, k, n);

        if (a[i_max*n+k] == 0) {
            printf("The matrix is singular!");
            return -1;
        }

        swap(a, c, k, i_max, k, n);

        printf("\nswaped k %d with i_max %d\n", k, i_max);
        printMatrix(a, n, n);
        printf("\n");

        printf("\nc:\n");
        printMatrix(c, n, 1);
        printf("\n");

        for (i = k+1; i < n; i++) {
            for (j = k+1; j < n; j++) {
                a[i*n+j] = a[i*n+j] - (a[i*n+k] / a[k*n+k]) * a[k*n+j];
            }
            c[i] = c[i] - (a[i*n+k] / a[k*n+k]) * c[k];
            a[i*n+k] = 0;
        }
    }

    // solve backward
    for (i = n-1; i >= 0; i--) {
        x[i] = c[i] / a[i*n+i];
        for (j = i+1; j < n; j++) {
            x[i] -= (a[i*n+j] * x[j]) / a[i*n+i];
        }
    }

    printf("x:\n");
    printMatrix(x, n, 1);
    printf("\n");

    return 0;
}
