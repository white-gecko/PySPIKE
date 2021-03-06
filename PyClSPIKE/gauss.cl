typedef struct {
    __global float *mat;
    int m, n;
} buffer;

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
 * Access buffer
 */
float _a(buffer b, int i, int j)
{
    int gid = get_global_id(0);
    return b.mat[(gid*b.m+i)*b.n+j];
}

/**
 * set value in buffer
 */
void _s(buffer b, int i, int j, float value)
{
    int gid = get_global_id(0);
    b.mat[(gid*b.m+i)*b.n+j] = value;
}

typedef struct {
    buffer b;
    int i, j, k, l;
} matrix;

/**
 * X = A-(B·C)
 */
void multiplySubstractMatrix (matrix r, matrix a, matrix b, matrix c) {
    if (b.l-b.j != c.k-c.i) {
        int gid = get_global_id(0);
        return;
    }

    // TODO check dimensions of A if it is the same as B·C

    int i,j,k;
    float value;
        int gid = get_global_id(0);
    for (i = 0; i < b.k-b.i; i++) {
        for (j = 0; j < c.l-c.j; j++) {
            value = 0;
            for (k = 0; k < b.l-b.j; k++) {
                value += _a(b.b, i + b.i, k + b.j) * _a(c.b, k + c.i, j + c.j);
            }
            value = _a(a.b, i + a.i, j + a.j) - value;
            _s(r.b, i + r.i, j + r.j, value);
        }
    }
    return;
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
        //printf((__constant char *)"Partition (%d): The matrix has to be wider than high.\n", gid);
        // write error to error register
        return;
    }

    // forward elimination
    for (k = 0; k < m; k++) {
        if (a[_(k,k,m,n,gid)] == (float)0) {
            //printf((__constant char *)"Partition (%d): The matrix is singular at position k=%d (a[k,k]: %f)!\n", gid, k, a[_(k,k,m,n,gid)]);
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
__kernel void reconstruct(__global float *vwg, __global float *x, __global float *xo, __global float *tmp, int m, int offSize, int rhsSize)
{
    int gid = get_global_id(0);
    // extract A_j, B_j and C_j from row
    //a[gid*n+...]

    // offSize - offdiagonalSize
    // m - partitionSize
    int n = 2 * offSize + rhsSize;

    buffer vwgBuf = {vwg, m, n};
    buffer xBuf = {x, 2*offSize, rhsSize};
    // make sure offSize is equal to the #columns of RHS
    buffer xoBuf = {xo, m - (2 * offSize), rhsSize};
    buffer tmpBuf = {tmp, m - (2 * offSize), rhsSize};

    if (gid == 0) {
        // case X'1 = G'1 - V'1 * Xt2
        // GPU xo[0] = vwg[_(offSize,2*offSize,m,n,gid)] - v[_(offSize,0,m,n,gid)] * x[_(2*offSize,0,2*offSize,offSize,gid)];
        matrix vm  = {vwgBuf, offSize,     0,           m - offSize,       offSize};
        matrix gm  = {vwgBuf, offSize,     2 * offSize, m - offSize,       2 * offSize + rhsSize};
        matrix xtm = {xBuf,   2 * offSize, 0,           3 * offSize,       rhsSize};
        matrix xom = {xoBuf,  0,           0,           m - (2 * offSize), rhsSize};

        multiplySubstractMatrix(xom, gm, vm, xtm);
    } else if (gid < get_global_size(0)) {
        // case X'j
        //xo[gid] = g[gid] - v[gid]*x[gid+1 t] - w[gid]*x[gid-1 b];

        matrix vm  = {vwgBuf, offSize,     0,           m - offSize,       offSize};
        matrix wm  = {vwgBuf, offSize,     offSize,     m - offSize,       2 * offSize};
        matrix gm  = {vwgBuf, offSize,     2 * offSize, m - offSize,       2 * offSize + rhsSize};
        matrix xtm = {xBuf,   2 * offSize, 0,           3 * offSize,       rhsSize};
        matrix xbm = {xBuf,   -offSize,    0,           0,                 rhsSize};
        matrix tmp = {tmpBuf, 0,           0,           m - (2 * offSize), rhsSize};
        matrix xom = {xoBuf,  0,           0,           m - (2 * offSize), rhsSize};

        multiplySubstractMatrix(tmp, gm, vm, xtm);
        multiplySubstractMatrix(xom, tmp, wm, xbm);
    } else {
        // case X'p
        //xo[gid] = g[gid] - w[gid]*x[gid-1 b];
        matrix wm  = {vwgBuf, offSize,     offSize,     m - offSize,       2 * offSize};
        matrix gm  = {vwgBuf, offSize,     2 * offSize, m - offSize,       2 * offSize + rhsSize};
        matrix xbm = {xBuf,   -offSize,    0,           0,                 rhsSize};
        matrix xom = {xoBuf,  0,           0,           m - (2 * offSize), rhsSize};

        multiplySubstractMatrix(xom, gm, wm, xbm);
    }

    return;
}
