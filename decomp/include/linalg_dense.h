#ifndef __LINALG_DENSE_H__
#define __LINALG_DENSE_H__

//#include "benlib.h"
#include "linalg_sparse.h"

// pain in the arse LAPACK routines for which I can't seem to find a common sense c interface
extern int dgeevx_( char*, char*, char*, char*, int*, double*, int*, double*, double*, double*, int*, double*, int*, int*, int*, double*, double*, double*, double*, double*, int*, int*, int*  );
extern void dgeqrf_( int*, int*, double*, int*, double*, double*, int*, int* );
extern void dormqr_( char *, char*, int*, int*, int*, double*, int*, double*, double *, int*, double*, int*, int* );
extern int dgesv_( int*, int*, double*, int*, int*, double*, int*, int* );
extern void dorgqr_( int*, int*, int*, double*, int*, double*, double*, int*, int* );
extern void dgetrf_( int *m, int *n, double *a, int *lda, int *ipiv, int *info );
extern void dgetri_( int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info );

typedef struct mtx
{
    int init;
    int nrows;
    int ncols;
    double *dat;
    double *tau;        // used to store the output of the same name from LAPACL least squares routines
} Tmtx, *Tmtx_ptr;

extern void mtx_init( Tmtx_ptr A, int nrows, int ncols );
extern void mtx_free( Tmtx_ptr A );
extern void mtx_copy( Tmtx_ptr from, Tmtx_ptr to );
extern void mtx_print( FILE *stream, Tmtx_ptr A );
extern void mtx_clear( Tmtx_ptr A );
extern void mtx_givens( Tmtx_ptr A, int k, double *rhs );
extern void vec_print( FILE *stream, double *x, int n );
extern int mtx_eigs( Tmtx_ptr A, double *lambdar, double *lambdai, double *v );
extern void mtx_QR_extractQ( Tmtx_ptr A );
extern void mtx_QR( Tmtx_ptr A );
extern int vec_drop_block( double* v, int *indx, int n, int lfill, int diag, double tol, double *u );
extern int vec_drop( double* v, int *indx, int n, int lfill, int diag, double tol );

#endif
