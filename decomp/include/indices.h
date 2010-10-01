/*
 *  indices.h
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef __INDICES_H__
#define __INDICES_H__

typedef struct index
{
	int dim_major;
	int dim_minor;
	int *index_major;
	int *index_minor;
} Tindex, *Tindex_ptr;

extern void index_init( Tindex_ptr index, int dim_major, int dim_minor );
extern void index_free( Tindex_ptr index );
extern void index_split( int *index_major, int *index_minor, int *index_dist, int dim_major, int dim_minor, int n_dom, Tindex_ptr split, int normalise );
extern void index_print( FILE *stream, Tindex_ptr index );
extern void index_CRS_to_CCS( int *cindxCRS, int *rindxCRS, int *cindxCCS, int *rindxCCS, int *p, int nnz, int nrows, int ncols );
extern int index_cmp( Tindex_ptr index1, Tindex_ptr index2 );
extern int index_verify( FILE *fid, Tindex_ptr index, int max_major );
extern void index_CRS_to_CCS_findperm( int *cindx, int *rindx, int *perm, int *rindxCCS, long long *order, int nrows, int nnz );
extern void index_make_adjacency( int n_nodes, int diag_shift, int *cindx, int *rindx, int *_cindx, int *_rindx );

#endif
