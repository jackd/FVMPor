/*
	libalg_dense_mpi.h
*/

#ifndef __LINALG_DENSE_MPI_H__
#define __LINALG_DENSE_MPI_H__

#include "ben_mpi.h"
#include "linalg.h"

// distributed dense matrix type
typedef struct mtx_dist
{
	int init;
	int block;
	TMPI_dat This;
	int *vtxdist;
	int *vtxdist_b;
	int *vtxspace;
	Tmtx mtx;
} Tmtx_dist, *Tmtx_dist_ptr;

// distributed dense vector type
typedef struct vec_dist
{
	int init;
	int block;
	int n;
	TMPI_dat This;
	int *vtxdist;
	int *vtxdist_b;
	int *vtxspace;
	double *dat;
} Tvec_dist, *Tvec_dist_ptr;

// distributed matrix routines
extern void mtx_dist_init( Tmtx_dist_ptr A, TMPI_dat_ptr This, int nrows, int ncols, int block, int *vtxdist );
extern void mtx_dist_init_explicit( Tmtx_dist_ptr A, TMPI_dat_ptr This, int nrows, int ncols, int block, int *vtxdist );
extern void mtx_dist_free( Tmtx_dist_ptr A );
extern void mtx_dist_clear( Tmtx_dist_ptr A );
extern void mtx_dist_gather( Tmtx_dist_ptr Alocal, Tmtx_ptr Aglobal, int root );
extern void mtx_dist_print( FILE *fid, Tmtx_dist_ptr A, int root );

// distributed vector routines
extern void vec_dist_init( Tvec_dist_ptr x, TMPI_dat_ptr This, int n, int block, int *vtxdist );
extern void vec_dist_init_vec( Tvec_dist_ptr x, Tvec_dist_ptr y );
extern void vec_dist_free( Tvec_dist_ptr x );
extern void vec_dist_print( FILE *fid, Tvec_dist_ptr x, int root );
extern void vec_dist_gather( Tvec_dist_ptr xlocal, double *xglobal, int root );
extern void vec_dist_gather_all( Tvec_dist_ptr xlocal, double *xglobal );
extern void vec_dist_scatter( Tvec_dist_ptr x_dist, double *x, TMPI_dat_ptr This, int block, int n, int root );
extern void vec_dist_scatter_vec( Tvec_dist_ptr x, double *x_local, int root, Tvec_dist_ptr y );
extern void vec_MPI_add( double *x, int n, TMPI_dat_ptr This );
extern void vec_dist_clear( Tvec_dist_ptr x );
extern void vec_dist_local_scatter( Tvec_dist_ptr x_dist, double *x, TMPI_dat_ptr This, int block, int n );

// Level 1 style vector-vector operations
extern void vec_dist_axpy( Tvec_dist_ptr x, Tvec_dist_ptr y, double alpha );
extern void vec_dist_copy( Tvec_dist_ptr x, Tvec_dist_ptr y );
extern void vec_dist_scale( Tvec_dist_ptr x, double alpha );
extern double vec_dist_nrm2( Tvec_dist_ptr x );
extern double vec_dist_normalise( Tvec_dist_ptr x );
extern double vec_dist_dot( Tvec_dist_ptr x, Tvec_dist_ptr y );

// Level 2 style mixed vector-matrix operations
extern void mtx_dist_insert_col( Tmtx_dist_ptr A, int col, Tvec_dist_ptr x );
extern void mtx_dist_get_col( Tmtx_dist_ptr A, int col, Tvec_dist_ptr x );
extern void mtx_dist_gemv( Tmtx_dist_ptr A, Tvec_dist_ptr x, Tvec_dist_ptr y, double alpha, double beta, int r0, int r1, int c0, int c1, char trans, double *yglobal );

// Level 3 style matrix-matrix operations

#endif
