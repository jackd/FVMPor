#ifndef __LINALG_SPARSE_H__
#define __LINALG_SPARSE_H__

#include "benlib.h"
#include "indices.h"

/*
    definitions of constants
*/

#define CRS_TO_CCS 0
#define CCS_TO_CRS 1

/* 
    definitions of data types 
*/

// data types for storing a simple Yale sparse format matrix.
// both CCS and CRS matrices are stored in the same structural manner,
// so first define a yale structure, then cast it to both types.
typedef struct mtx_yale
{
    int block;      // flags if this is a block matrix
    int init;       // flags if this matrix has been intialised
    int nrows;      // number of rows
    int ncols;      // number of columns
    int nnz;        // number of nonzero elements
    int *cindx;     // an array of column indices
    int *rindx;     // an array of row indices
    double *nz;     // array of nonzero elements
} Tmtx_yale, *Tmtx_yale_ptr;

// compressed column storage
typedef Tmtx_yale Tmtx_CCS;
typedef Tmtx_yale_ptr Tmtx_CCS_ptr;

// compressed row storage
typedef Tmtx_yale Tmtx_CRS;
typedef Tmtx_yale_ptr Tmtx_CRS_ptr;

// define a sparse vector type
typedef struct vec_sp
{
    int init;
    int block;
    int n;
    int nnz;
    int *indx;
    double *nz;
} Tvec_sp, *Tvec_sp_ptr;

/*
    function prototypes
*/
// etc stuff
extern int dense_gather_lfill( double *x, double *y, int *p, int n, int lfill );

// general yale format routines
extern void mtx_yale_index_swap( Tmtx_yale_ptr A );     // used in transpose/CCS<->CRS operations
extern void mtx_CRS_to_CCS( Tmtx_CRS_ptr A, Tmtx_CCS_ptr B );
extern void mtx_CCS_to_CRS( Tmtx_CCS_ptr A, Tmtx_CRS_ptr B );

// CRS routines
extern void mtx_CRS_init( Tmtx_CRS_ptr A, int nrows, int ncols, int nnz, int block );
extern void mtx_CRS_copy( Tmtx_CRS_ptr from, Tmtx_CRS_ptr to );
extern void mtx_CRS_free( Tmtx_CRS_ptr A );
extern int mtx_CRS_equal( Tmtx_CRS_ptr A, Tmtx_CRS_ptr B );
extern void mtx_CRS_transpose( Tmtx_CRS_ptr A, Tmtx_CRS_ptr B );
extern void mtx_CRS_print( FILE *fid, Tmtx_CRS_ptr A );
extern int mtx_CRS_validate( Tmtx_CRS_ptr A );
extern void mtx_CRS_gemv( Tmtx_CRS_ptr A, double *x, double *y, double alpha, double beta, char trans );
extern void mtx_CRS_col_scale( Tmtx_CRS_ptr A, double *scales, int op );
extern void mtx_CRS_col_normalise( Tmtx_CRS_ptr A, double *norms );
extern void mtx_CRS_col_norms( Tmtx_CRS_ptr A, double *norms );

// CCS routines
extern int  mtx_CCS_is_connected( Tmtx_CCS_ptr A );
extern int  mtx_CCS_is_sym_structure( Tmtx_CCS_ptr A );
extern int  mtx_CCS_is_diag_nz( Tmtx_CCS_ptr A );
extern void mtx_CCS_transpose( Tmtx_CCS_ptr A, Tmtx_CCS_ptr B );
extern void mtx_CCS_init( Tmtx_CCS_ptr A, int nrows, int ncols, int nnz, int block );
extern void mtx_CCS_copy( Tmtx_CCS_ptr from, Tmtx_CCS_ptr to );
extern void mtx_CCS_free( Tmtx_CRS_ptr A );
extern void mtx_CCS_print( FILE *fid, Tmtx_CCS_ptr A );
extern void mtx_CCS_axpy( double a, Tmtx_CCS_ptr X, Tmtx_CCS_ptr Y );
extern void mtx_CCSB_axpy( double a, Tmtx_CCS_ptr X, Tmtx_CCS_ptr Y );
extern void mtx_CCS_mat_vec_mult( Tmtx_CCS_ptr A, double *x, double *y, int yzero );
extern void mtx_CCS_getcol_sp( Tmtx_CCS_ptr A, Tvec_sp_ptr x, int col );
extern void mtx_CCS_eye( Tmtx_CCS_ptr A, int dim, int block );
extern int mtx_CCS_validate( Tmtx_CCS_ptr A );
extern void mtx_CCS_column_unpack_block( Tmtx_CCS_ptr A, double *v, int col );
extern void mtx_CCS_column_add_unpacked_block( Tmtx_CCS_ptr A, double *u, int col, double alpha, double beta );

// sparse vector routines
extern void vec_sp_init( Tvec_sp_ptr v, int n, int nnz, int block );
extern void vec_sp_free( Tvec_sp_ptr x );
extern void vec_sp_axpy( double a, Tvec_sp_ptr x, Tvec_sp_ptr y );
extern void vec_sp_print( FILE *fid, Tvec_sp_ptr x );
extern void vec_sp_copy( Tvec_sp_ptr x, Tvec_sp_ptr y );
extern double vec_sp_ip( Tvec_sp_ptr x, Tvec_sp_ptr y );
extern double vec_sp_nrm2( Tvec_sp_ptr x );
extern void vec_sp_scatter( Tvec_sp_ptr x, double *y );
extern void vec_sp_gather( Tvec_sp_ptr y, double *x, int n, int block );
extern void vec_sp_gather_lfill( Tvec_sp_ptr y, double *x, int n, int block, int lfill );

// sparse-sparse matrix vector routines
extern void mtx_CRS_gemv_sss( Tmtx_CRS_ptr A, Tvec_sp_ptr x, Tvec_sp_ptr y );
extern void mtx_CRS_gemv_ssd( Tmtx_CRS_ptr A, Tvec_sp_ptr x, double *y );
extern void mtx_CRS_gemv_sdd( Tmtx_CRS_ptr A, double *x, double *y );
extern void mtx_CRS_vec_mul( Tmtx_CRS_ptr A, double *x, double *y );

#endif

