/*
	linalg_mpi.h
*/
#ifndef __LINALG_MPI_H__
#define __LINALG_MPI_H__

#include "linalg.h"
#include "ben_mpi.h"
#include "linalg_dense_mpi.h"
#include "indices.h"

// holds the domain information for a domain that has been decomposed
// using Metis and stored appropriately
typedef struct domain
{
	int n_dom;		// number of domains in decomposition
	int n_node;		// total number of nodes in the domain
	int *vtxdist;   // array holding details of which processors have which points
	int *map;		// map of who is connected to who
	int *n_in;		// number of internal nodes in each domain
	int *n_bnd;		// the number of boundary nodes in each domain
} Tdomain, *Tdomain_ptr;

// information used in a split
typedef struct split
{
	// global split information
	TMPI_dat This;
	Tdomain dom;
	int *counts;
	
	// local split information
	int n_int;
	int n_bnd;
	int n_ext;
	int *nodes_ext;	
} Tsplit, *Tsplit_ptr;

typedef struct distribution
{
	int n_dom;
	int n_node;
	int n_neigh;
	int *neighbours;
	int *counts;
	int *starts;
	int *indx;
	int *part;
	int *ppart;
} Tdistribution, *Tdistribution_ptr;

typedef struct mtx_CRS_dist
{
	// these params describe the locally stored matrix
	Tmtx_CRS mtx;
	
	// the following params pertain to the distributed parts of the matrix
	int init;
	int nrows;		// number of rows in distributed matrix
	int ncols;		// number of columns
	int *vtxdist;
	TMPI_dat This;  // communicator information for This matrix
} Tmtx_CRS_dist, *Tmtx_CRS_dist_ptr;

/*
	defines a CSR matrix split into four blocks. The matrix is concerned with
	the nodes inside a domain, and their relationships. the 4 split is due
	to those nodes being split into two groups : internal and boundary. Thus
	we have internal-internal (B), internal-boundary (F), boundary-internal (E) and
	boundary-boundary (C) relationships. The Eij matrices relate to the relation
	between boundary points in domain i and boundary nodes in domain j.
*/
typedef struct mtx_CRS_split
{
	int init;
	int n;
	int n_dom; 
	int this_dom;
	int n_in;
	int n_bnd;
	int n_neigh;
	int *neighbours;
	Tmtx_CRS B;			// the four block matrices in CRS format
	Tmtx_CRS C;
	Tmtx_CRS E;
	Tmtx_CRS F;
	Tmtx_CRS_ptr Eij;   // the connection matrices
} Tmtx_CRS_split, *Tmtx_CRS_split_ptr;

//extern void METIS_PartGraphKway( int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );

extern void mtx_CRS_dist_init( Tmtx_CRS_dist_ptr A, int nrows, int ncols, int nnz, int block, TMPI_dat_ptr This );
extern void mtx_CRS_dist_copy( Tmtx_CRS_dist_ptr from, Tmtx_CRS_dist_ptr to );
extern void mtx_CRS_dist_free( Tmtx_CRS_dist_ptr A );
extern void mtx_CRS_distribute( Tmtx_CRS_ptr A, Tmtx_CRS_dist_ptr Ad, TMPI_dat_ptr This, int src );
extern void mtx_CRS_dist_domdec( Tmtx_CRS_dist_ptr A, Tdistribution_ptr dist, int *p, int fillA );
extern void mtx_CRS_dist_domdec_sym( Tmtx_CRS_dist_ptr A, Tdistribution_ptr dist, int *p );
extern void mtx_CRS_dist_gemv( Tmtx_CRS_dist_ptr A, Tvec_dist_ptr x, Tvec_dist_ptr y, double alpha, double beta, char trans );
extern double mtx_CRS_dist_residual( Tmtx_CRS_dist_ptr A, Tvec_dist_ptr x, Tvec_dist_ptr b, Tvec_dist_ptr r );
extern void mtx_CRS_dist_col_normalise( Tmtx_CRS_dist_ptr A, double *norms );
extern void mtx_CRS_dist_gather( Tmtx_CRS_dist_ptr A, Tmtx_CRS_ptr Alocal, int root );

extern void distribution_init( Tdistribution_ptr dist, int n_dom, int n_node, int n_neigh );
extern void distribution_free( Tdistribution_ptr dist );

extern void index_map_find( Tmtx_CRS_dist_ptr A, int **cindx, int **rindx, int *dim_major, int insert );

extern void domain_init( Tdomain_ptr dom, TMPI_dat_ptr This, int n_node );
extern void domain_free( Tdomain_ptr dom );
extern void domain_copy( Tdomain_ptr from, Tdomain_ptr to );

extern void mtx_CRS_split_init( Tmtx_CRS_split_ptr A, int n, int n_in, int n_bnd, int n_dom, int this_dom, int n_neigh );
extern void mtx_CRS_split_free( Tmtx_CRS_split_ptr A );


#endif
