#include<stdio.h>

/*
 * code written according to:
 * http://de.wikipedia.org/wiki/Gau%C3%9Fsches_Eliminationsverfahren#Algorithmus_in_Pseudocode
 */

int main()
{
    int i,j,k,n = 3;
    float x[3];

    float a[3][3] = {{7,3,-5},{-1,-2,4},{-4,1,-3}};
    float c[] = {-12,5,1};

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f\t", a[i][j]);
        }
        printf("\n");
    }

    printf("\n");

    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            for (k = 0; k < i-1; k++) {
                a[i][j] -= a[i][k] * a[k][j];
            }
        }

        for (j = i+1; j < n; j++) {
            for (k = 0; k < i-1; k++) {
                a[j][i] -= a[j][k] * a[k][i];
            }

            a[j][i] = a[j][i] / a[i][i];
        }
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f\t", a[i][j]);
        }
        printf("\n");
    }
}
