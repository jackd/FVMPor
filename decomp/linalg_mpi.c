/*
	linalg_mpi.c
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "ben_mpi.h"
#include "linalg_mpi.h"
#include "linalg_dense_mpi.h"
#include "parmetis.h"
#include "linalg.h"
#include "indices.h"
#include "fileio.h"

void profile_output( int *cindx, int *rindx, int nrows, int ncols, int nnz, char *fname );

/******************************************************************************************
 *
 *  routine makes blocking MPI calls
 * 
******************************************************************************************/
void mtx_CRS_dist_init( Tmtx_CRS_dist_ptr A, int nrows, int ncols, int nnz, int block, TMPI_dat_ptr This )
{
	int i;
	
	// free the matrix if it has been previously initialised
	if( A->init )
		mtx_CRS_dist_free(A);
	
	if( !A->init )
		A->mtx.init = 0;
	mtx_CRS_init( &A->mtx, nrows, ncols, nnz, block );
	A->init = 1;
	A->ncols = ncols;
	
	// determine and store the number of rows in the global matrix
	MPI_Allreduce( &nrows, &A->nrows, 1, MPI_INT, MPI_SUM, This->comm );
	
	// find the vtxdist vector
	A->vtxdist = (int *)malloc( sizeof(int)*(This->n_proc+1) );
	MPI_Allgather( &nrows, 1, MPI_INT, A->vtxdist+1, 1, MPI_INT, This->comm );
	
	// convert the list into a vtxdist style distribution profile
	A->vtxdist[0] = 0;
	for( i=2; i<(This->n_proc+1); i++ )
		A->vtxdist[i] += A->vtxdist[i-1];
	BMPI_copy( This, &A->This );
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_CRS_dist_copy( Tmtx_CRS_dist_ptr from, Tmtx_CRS_dist_ptr to )
{
	if( !from->init )
	{
		fprintf( stderr, "mtx_CRS_dist_copy() : the source matrix from is not initialised" );
		MPI_Finalize();
		exit(1);
	}
	mtx_CRS_init( &to->mtx, from->mtx.nrows, from->mtx.ncols, from->mtx.nnz, from->mtx.block );
	mtx_CRS_copy( &from->mtx, &to->mtx );
	to->init = from->init;
	to->ncols = from->ncols;
	to->nrows = from->nrows;
	to->vtxdist = (int *)malloc( sizeof(int)*(from->This.n_proc+1) );
	BMPI_copy( &from->This, &to->This );
	return;
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_CRS_dist_free( Tmtx_CRS_dist_ptr A )
{
	if( !A->init )
	{
		fprintf( stderr, "mtx_CRS_dist_free() : cannot free an un_initialised matrix\n" );
		MPI_Finalize();
		exit(1);
	}
	mtx_CRS_free( &A->mtx );
	free( A->vtxdist );
	A->init = 0;
	BMPI_free( &A->This );
}

/******************************************************************************************
*
*	HAS A BUG. I don't have time to find it.
*
******************************************************************************************/
void mtx_CRS_dist_col_normalise( Tmtx_CRS_dist_ptr A, double *norms )
{
	int i, ncols;
	double *norms_temp;
	
	if( !A->init )
	{
		fprintf( stderr, "mtx_CRS_dist_col_normalise() : passed an uninitialised matrix" );
		MPI_Finalize();
		exit(1);
	}
	
	// initialise
	ncols = A->ncols;
	norms_temp = (double*)malloc( sizeof(double)*ncols );
	if( A->mtx.block )
		ncols = ncols BLOCK_V_SHIFT;
	
	// find the norms of the local columns
	mtx_CRS_col_norms( &A->mtx, norms );
	for( i=0; i<ncols; i++ )
		norms[i] *= norms[i];
	
	// find the sum of the squares of the norms for each column over all Pid
	MPI_Barrier(A->This.comm);
	MPI_Allreduce( norms, norms_temp, ncols, MPI_DOUBLE, MPI_SUM, A->This.comm );
		
	// find the column norms by taking sqare roots
	for( i=0; i<ncols; i++ )
		norms[i] = sqrt(norms_temp[i]);
	
	// scale the local columns by the norms
	//mtx_CRS_col_scale( &A->mtx, norms, 0 );
	
	free( norms_temp );
}


/*******************************************************************************************
 *	distributes a CRS matrix stored on processor src to all processors
 *
*******************************************************************************************/
void mtx_CRS_distribute( Tmtx_CRS_ptr A, Tmtx_CRS_dist_ptr Ad, TMPI_dat_ptr This, int src )
{
	int nrows, pos, dist_size, nnz, i, j, pattern;
	int my_nrows;
	int *vtxdist;
	MPI_Status status;
	
	// check that the user has passed a valid process number for src
	ASSERT_MSG( src>-1 && src<=This->n_proc, "mtx_CRS_distribute() : invalid src" );
	

// BUGFIX 2006
// -----------

// moved inside else clause

	// allocate some memory
//	vtxdist = (int *)malloc( sizeof(int)*(This->n_proc+1) );

// -----------

	// determine if This is a pattern matrix
	if( This->this_proc==src )
	{
		if( A->nz )
			pattern = 0;
		else
			pattern = 1;
	}
	MPI_Bcast( &pattern, 1, MPI_INT, src, This->comm );
	
	// distribute over a 1 process area?
	if( This->n_proc==1 )
	{
		nnz = A->nnz;

		mtx_CRS_dist_init( Ad, A->nrows, A->ncols, nnz, A->block, This );
		if( !pattern )
		{
			if( A->block )
				memcpy( Ad->mtx.nz, A->nz, sizeof(double)*(nnz BLOCK_M_SHIFT));
			else
				memcpy( Ad->mtx.nz, A->nz, sizeof(double)*nnz );	
		}
			
		memcpy( Ad->mtx.cindx, A->cindx, sizeof(int)*A->nnz );
		memcpy( Ad->mtx.rindx, A->rindx, sizeof(int)*(A->nrows+1) );
	}
	else
	{	

// BUGFIX 2006
// -----------
// vtxdist only allocated when necessary
	vtxdist = (int *)malloc( sizeof(int)*(This->n_proc+1) );
// -----------

		// are we the source?
		if( This->this_proc == src )
		{
			int block;
			
			// first check that a valid matrix has been passed
			if( (A==NULL) || (!A->init) )
			{
				BMPI_isok( This, 0 );
				fprintf( stderr, "[master] mtx_CRS_distribute() : invalid matrix passed for distribution\n" );
				MPI_Finalize();
				exit(1);
			}
			else
			{
				BMPI_isok( This, 1 );
			}
			
			// initialise
			block = A->block;
			
			// determine who shall get which rows
			nrows = A->nrows;
			dist_size = floor( ((double)nrows)/This->n_proc );
			
			vtxdist[0] = 0;
			for( i=0, pos=0; i<nrows - (dist_size*This->n_proc); i++ )
			{
				pos += (dist_size+1);
				vtxdist[i+1] = pos;
			}
			for( ; i<This->n_proc; i++ )
			{
				pos += dist_size;
				vtxdist[i+1] = pos;
			}

		
			// broadcast the number of rows in the matrx
			MPI_Bcast( &A->nrows, 1, MPI_INT, src, This->comm );
			MPI_Bcast( &A->ncols, 1, MPI_INT, src, This->comm );
			MPI_Bcast( &A->block, 1, MPI_INT, src, This->comm );
			MPI_Bcast( vtxdist, This->n_proc+1, MPI_INT, src, This->comm );
			MPI_Barrier( This->comm );

		
			// send each process its nnz count and intialise the distributed type
			for( i=0; i<This->n_proc; i++ )
			{
				if( i!=src )
				{
					nnz = A->rindx[vtxdist[i+1]] - A->rindx[vtxdist[i]];
					MPI_Send( &nnz, 1, MPI_INT, i, src, This->comm );
				}
			}
			
			// send out the matrix parts
			for( i=0; i<This->n_proc; i++ )
			{
				// only send out the rows if they are not This process's
				if( i!=src )
				{				
					// send the chunk
					nnz = A->rindx[vtxdist[i+1]] - A->rindx[vtxdist[i]];
					if( !pattern )
					{
						if( !block )
							MPI_Send( A->nz + A->rindx[vtxdist[i]], nnz, MPI_DOUBLE, i, src, This->comm );
						else 
							MPI_Send( A->nz + (A->rindx[vtxdist[i]] BLOCK_M_SHIFT), (nnz BLOCK_M_SHIFT), MPI_DOUBLE, i, src, This->comm );
					}
					MPI_Send( A->cindx + A->rindx[vtxdist[i]], nnz, MPI_INT, i, src, This->comm );
					MPI_Send( A->rindx + vtxdist[i], vtxdist[i+1]-vtxdist[i]+1, MPI_INT, i, src, This->comm );
				}
				// This is my own chunk
				else
				{
					my_nrows = vtxdist[i+1]-vtxdist[i];
					nnz = A->rindx[vtxdist[i+1]] - A->rindx[vtxdist[i]];
					mtx_CRS_dist_init( Ad, my_nrows, A->ncols, nnz, A->block, This );
					if( !pattern )
					{
						if( !block )
							memcpy( Ad->mtx.nz, A->nz + A->rindx[vtxdist[i]],nnz*sizeof(double) );
						else
							memcpy( Ad->mtx.nz, A->nz + (A->rindx[vtxdist[i]] BLOCK_M_SHIFT), (nnz*sizeof(double))BLOCK_M_SHIFT );
					}
					memcpy( Ad->mtx.cindx, A->cindx + A->rindx[vtxdist[i]], nnz*sizeof(int) );
					memcpy( Ad->mtx.rindx, A->rindx + vtxdist[i], (my_nrows+1)*sizeof(int) );
					pos = Ad->mtx.rindx[0];
					for( j=0; j<my_nrows+1; j++ )
					{
						Ad->mtx.rindx[j] -= pos;
					}
					free( Ad->vtxdist );
					Ad->vtxdist = vtxdist;
				}
			}
		}
		
		// otherwise we are one of the destinations
		else
		{
			int mtx_nrows;
			int mtx_ncols;
			int block;
			
			// get the thumbs up from the source process
			if( !BMPI_ok( This, src ) )
			{
				MPI_Finalize();
				MPI_Barrier( This->comm );
				MPI_Abort( This->comm, 0 );
			}
			
			// recieve broadcast information about the matrix
			MPI_Bcast( &mtx_nrows, 1, MPI_INT, src, This->comm );
			MPI_Bcast( &mtx_ncols, 1, MPI_INT, src, This->comm );
			MPI_Bcast( &block, 1, MPI_INT, src, This->comm );
			MPI_Bcast( vtxdist, This->n_proc+1, MPI_INT, src, This->comm );
			MPI_Barrier( This->comm );
			
			
			// determine how many rows are going to be in local chunk
			my_nrows = vtxdist[This->this_proc+1]-vtxdist[This->this_proc];
			
			// recieve specific details of This matrix size
			MPI_Recv( &nnz, 1, MPI_INT, src, src, This->comm, &status );
			
			// initialise the memory and data type for the local chunk
			mtx_CRS_dist_init( Ad, my_nrows, mtx_ncols, nnz, block, This );
			
			// recieve the matrix data
			if( !pattern )
			{
				if( !block )
					MPI_Recv( Ad->mtx.nz, nnz, MPI_DOUBLE, src, src, This->comm, &status );
				else
					MPI_Recv( Ad->mtx.nz, (nnz BLOCK_M_SHIFT), MPI_DOUBLE, src, src, This->comm, &status );
			}
			MPI_Recv( Ad->mtx.cindx, nnz, MPI_INT, src, src, This->comm, &status );
			MPI_Recv( Ad->mtx.rindx, my_nrows+1, MPI_INT, src, src, This->comm, &status );
			
			
			// localise the matrix data
			pos = Ad->mtx.rindx[0];
			for( i=0; i<my_nrows+1; i++ )
				Ad->mtx.rindx[i] -= pos;
			
			// and finally hook the correct vtxdist up
			free( Ad->vtxdist );
			Ad->vtxdist = vtxdist;
		}
		
	}


	// free up the nz data if This is a pattern matrix
	if( pattern )
	{
		free( Ad->mtx.nz );
		Ad->mtx.nz = NULL;
	}

}

/*******************************************************************************************
 *	mtx_CRS_dist_domdec( )
 *	
 *		Find the partition for the graph using ParMETIS
 *
 *		The domain that each node stored on This process belongs to is
 *		found using ParMETIS and stored in part. This array is then sorted
 *		to find which domains we have to send nodes to, and how many nodes
 *		they require from us. The array part ends up containing a sorted version
 *		of part returned by ParMETIS, and indx contains the permutaion array
 *		that performed the sort. ppart contains the original partitiion array, stored
 *		in the natural node order. counts[i] stores how many nodes from domain i are
 *		on This process, and starts[i]->starts[i+1]-1 are the indices into indx for those
 *		nodes.
*******************************************************************************************/
void mtx_CRS_dist_domdec( Tmtx_CRS_dist_ptr A, Tdistribution_ptr dist, int *p, int fillA )
{
	int i, n_nodes, dom, count, n_dom, pos, edgecut, numflag=0, wgtflag=0, this_dom, n_neigh=0;
	int *vstarts=NULL, *indx=NULL, *starts=NULL, *counts=NULL, *part=NULL, *ppart=NULL, *_cindx=NULL, *_rindx=NULL;
	int *cindx=NULL, *rindx=NULL, dim=0;
	int ParMETIS_options[4] = {0, 0, 0, 0};
	TMPI_dat_ptr This=NULL;
	
	This = &A->This;
	
	// initialise variables
	n_nodes  =  A->mtx.nrows;
	n_dom    =  This->n_proc;
	this_dom =  This->this_proc;
	
	// initialise the distribution
	distribution_init( dist, n_dom, n_nodes, 1 );
	
	// setup pointers
	indx   = dist->indx;
	starts = dist->starts;
	counts = dist->counts;
	part   = dist->part;
	ppart  = dist->ppart;

	// find the symmetric pattern
	index_map_find( A, &cindx, &rindx, &dim, fillA );

	// make the CRS profile of the matrix into an adjacency graph
	// later This can be incorporated into index_map_find()
	_cindx = malloc( (dim)*sizeof(int) );      
	_rindx = malloc( (n_nodes+1)*sizeof(int) );
	//index_make_adjacency( n_nodes, A->vtxdist[this_dom], cindx, rindx, _cindx, _rindx );
	index_make_adjacency( n_nodes, A->vtxdist[this_dom], A->mtx.cindx, A->mtx.rindx, _cindx, _rindx );
	
	// use ParMETIS to perform domain decomp
	ParMETIS_PartKway( A->vtxdist, _rindx, _cindx, NULL, NULL, &wgtflag, &numflag, &n_dom, 
					   ParMETIS_options, &edgecut, part, &This->comm);
		
	// keep copy of original part in ppart
	memcpy( ppart, part, sizeof(int)*n_nodes );
	
	// sort part, indx holds the permutation required for This
	for( i=0; i<n_nodes; i++ ) 
		indx[i]=i;
	heapsort_int_index( n_nodes, indx, part);
	
	// determine the number of nodes that we have for each processor
	pos = 0;
	for( dom=0; dom<n_dom; dom++ )
	{
		starts[dom] = pos;
		count=0;

// BUGFIX 2006
// -----------
//		while( part[pos]==dom && pos<A->mtx.nrows ) 
		while( pos<A->mtx.nrows && part[pos]==dom ) 
// -----------

		{
			pos++;
			count++;
		}
		counts[dom] = count;
		if( count )
			n_neigh++;
	}
	starts[dom] = pos;
	
	// find and store the neighbour information to dist
	free( dist->neighbours );
	dist->neighbours = (int *)malloc( sizeof(int)*n_neigh );
	dist->n_neigh = n_neigh;
	for( pos=0, dom=0; dom<n_dom; dom++ )
		if( counts[dom] )
			dist->neighbours[pos++] = dom;
	
	
	// sort the indices of each target domain's nodes
	for( dom=0; dom<n_dom; dom++ )
		if( counts[dom]>1 )
			heapsort_int( counts[dom], indx + starts[dom] );
	
	// all processes update their copy of the global partition vector in p
	vstarts = (int *)malloc( sizeof(int)*n_dom );
	for( i=0; i<n_dom; i++ )
		vstarts[i] = A->vtxdist[i+1] - A->vtxdist[i];
	
	MPI_Allgatherv( ppart, vstarts[this_dom], MPI_INT, p, vstarts, A->vtxdist, MPI_INT, This->comm );
	free( vstarts );
	free( _rindx );
	free( _cindx );
	free( rindx );
	free( cindx );
}

// same as above, but assumes that the matrix A has symmetric sparsity pattern. Only recommended for use
// on a matrix that has been formed directly from a finite volume mesh.
void mtx_CRS_dist_domdec_sym( Tmtx_CRS_dist_ptr A, Tdistribution_ptr dist, int *p )
{
	int i, n_nodes, dom, count, n_dom, pos, edgecut, numflag=0, wgtflag=0, this_dom, n_neigh=0;
	int *vstarts=NULL, *indx=NULL, *starts=NULL, *counts=NULL, *part=NULL, *ppart=NULL, *_cindx=NULL, *_rindx=NULL;
	int ParMETIS_options[4] = {0, 0, 0, 0};
	TMPI_dat_ptr This=NULL;
	
	This = &A->This;
	
	// initialise variables
	n_nodes  =  A->mtx.nrows;
	n_dom    =  This->n_proc;
	this_dom =  This->this_proc;
	
	// initialise the distribution
	distribution_init( dist, n_dom, n_nodes, 1 );
	
	// setup pointers
	indx   = dist->indx;
	starts = dist->starts;
	counts = dist->counts;
	part   = dist->part;
	ppart  = dist->ppart;

	// make the CRS profile of the matrix into an adjacency graph
	_cindx = malloc( (A->mtx.nnz)*sizeof(int) );
	_rindx = malloc( (n_nodes+1)*sizeof(int) );
	index_make_adjacency( n_nodes, A->vtxdist[this_dom], A->mtx.cindx, A->mtx.rindx, _cindx, _rindx );

	// use ParMETIS to perform domain decomp
	ParMETIS_PartKway( A->vtxdist, _rindx, _cindx, NULL, NULL, &wgtflag, &numflag, &n_dom, 
					   ParMETIS_options, &edgecut, part, &This->comm);
		
	// keep copy of original part in ppart
	memcpy( ppart, part, sizeof(int)*n_nodes );
	
	// sort part, indx holds the permutation required for This
	for( i=0; i<n_nodes; i++ ) 
		indx[i]=i;
	heapsort_int_index( n_nodes, indx, part);
	
	// determine the number of nodes that we have for each processor
	pos = 0;
	for( dom=0; dom<n_dom; dom++ )
	{
		starts[dom] = pos;
		count=0;
		while( part[pos]==dom && pos<A->mtx.nrows ) 
		{
			pos++;
			count++;
		}
		counts[dom] = count;
		if( count )
			n_neigh++;
	}
	starts[dom] = pos;
	
	// find and store the neighbour information to dist
	free( dist->neighbours );
	dist->neighbours = (int *)malloc( sizeof(int)*n_neigh );
	dist->n_neigh = n_neigh;
	for( pos=0, dom=0; dom<n_dom; dom++ )
		if( counts[dom] )
			dist->neighbours[pos++] = dom;
	
	
	// sort the indices of each target domain's nodes
	for( dom=0; dom<n_dom; dom++ )
		if( counts[dom]>1 )
			heapsort_int( counts[dom], indx + starts[dom] );
	
	// all processes update their copy of the global partition vector in p
	vstarts = (int *)malloc( sizeof(int)*n_dom );
	for( i=0; i<n_dom; i++ )
		vstarts[i] = A->vtxdist[i+1] - A->vtxdist[i];
	
	MPI_Allgatherv( ppart, vstarts[this_dom], MPI_INT, p, vstarts, A->vtxdist, MPI_INT, This->comm );
	free( vstarts );
	free( _rindx );
	free( _cindx );
}

/******************************************************************************************
 *	void domain_init( Tdomain_ptr dom, TMPI_dat_ptr This )
 *
******************************************************************************************/
void domain_init( Tdomain_ptr dom, TMPI_dat_ptr This, int n_node )
{
	dom->n_dom   = This->n_proc;
	dom->n_node  = n_node;
	dom->vtxdist = (int *)malloc( sizeof(int)*(dom->n_dom+1) );
	dom->map     = (int *)malloc( sizeof(int)*dom->n_dom*dom->n_dom );
	dom->n_in    = (int *)malloc( sizeof(int)*dom->n_dom );
	dom->n_bnd   = (int *)malloc( sizeof(int)*dom->n_dom );
}

/******************************************************************************************
 *	void domain_free( Tdomain_ptr dom )
 *
******************************************************************************************/
void domain_free( Tdomain_ptr dom )
{
	dom->n_dom   = 0;
	dom->n_node  = 0;

// BUGFIX 2006
// -----------

// Tests (in this case) are harmless, but still unnecessary.  Search for BUGFIX
// to find examples of similar, but harmful tests in this code.

//	if( dom->vtxdist )
		free( dom->vtxdist );
//	if( dom->map )
		free( dom->map );
//	if( dom->n_in )
		free( dom->n_in );
//	if( dom->n_bnd )
		free( dom->n_bnd );

// -----------
	
	dom->vtxdist = NULL;
	dom->map     = NULL;
	dom->n_bnd   = NULL;
	dom->n_in    = NULL;
}

/******************************************************************************************
 *	distribution_init()
 *
 *	allocate memory for a distribution data structure
 *
 *	n_dom is the number of domains for the distribution
******************************************************************************************/
void distribution_init( Tdistribution_ptr dist, int n_dom, int n_node, int n_neigh )
{
	dist->n_dom   = n_dom;
	dist->n_node  = n_node;
	dist->n_neigh = n_neigh;
	
	dist->indx       = (int*)malloc( sizeof(int)*n_node );
	dist->counts	 = (int*)malloc( sizeof(int)*n_dom );
	dist->starts     = (int*)malloc( sizeof(int)*(n_dom+1) );
	dist->part       = (int*)malloc( sizeof(int)*n_node );
	dist->ppart      = (int*)malloc( sizeof(int)*n_node );
	dist->neighbours = (int*)malloc( sizeof(int)*n_neigh );
}

/******************************************************************************************
 *	distribution_free()
 *
 *	free memory for a distribution data structure
******************************************************************************************/
void distribution_free( Tdistribution_ptr dist )
{
	dist->n_dom = 0;
	dist->n_node = 0;
	
// BUGFIX 2006
// -----------

//	if( dist->indx   )
		free( dist->indx   );
//	if( dist->counts )
		free( dist->counts );
//	if( dist->starts )
		free( dist->starts );
//	if( dist->part   )
		free( dist->part   );
//	if( dist->ppart  )
		free( dist->ppart  );

    free (dist->neighbours );

// -----------
	
	dist->indx   = NULL;
	dist->counts = NULL;
	dist->starts = NULL;
	dist->part   = NULL;
	dist->ppart  = NULL;
}

/******************************************************************************************
*	index_map_find()
*
*	calculates the symmetric CRS sparsity pattern needed for ParMETIS. The sparsity pattern
*   returned in (cindx,rindx,dim_major), where dim_major=length(cindx), is symmetric.
*
*	insert flags the update of A. if update=0, only the new sparsity pattern is created 
*	and returned, with A unchanged. If update=1, A is updated to the new sparsity pattern, 
*	with zeros inserted at the new positions.
*	
******************************************************************************************/
void index_map_find( Tmtx_CRS_dist_ptr A, int **cindx, int **rindx, int *length, int insert )
{
	int i, j, k, spot, this_dom, nnz, nrows, ncols, nc, nr, n, pos, nd, td;
	Tindex split[3];
	int vtxdist[4];
	int *cindxCRS, *rindxCRS, *cindxCCS, *rindxCCS, *cp, *rp, *head, *index_minor, *index_major;
	
	// setup constants
	this_dom = A->This.this_proc;
	if( this_dom==0 )
	{
		nd = 2;
		td = 0;
	}
	else if( this_dom==(A->This.n_proc-1) )
	{	
		nd = 2;
		td = 1;
	}
	else
	{
		nd = 3;
		td = 1;
	}
		
	/*
	 *	split the index into 3 sections, left, centre (the S block), and right
	 */
	vtxdist[0] = 0;
	vtxdist[td] = A->vtxdist[this_dom];
	vtxdist[td+1] = A->vtxdist[this_dom+1];
	vtxdist[nd] = A->ncols;
	index_split( A->mtx.cindx, A->mtx.rindx, vtxdist, A->mtx.nnz, A->mtx.nrows, nd, split, 1 );
		
	/*
	 *	now make the centre index symmetric
	 */
	cindxCRS = split[td].index_major;
	rindxCRS = split[td].index_minor;
	nnz = split[td].dim_major;
	nrows = A->mtx.nrows;
	ncols = nrows;
	cindxCCS = (int *)malloc( sizeof(int)*(ncols+1) );
	rindxCCS = (int *)malloc( sizeof(int)*(nnz) );
	
	// convert the CRS index to CCS
	index_CRS_to_CCS( cindxCRS, rindxCRS, cindxCCS, rindxCCS, NULL, nnz, nrows, ncols );
	// swap the column and row indices over for the CCS to make them "CRS"
	cp = cindxCCS;
	cindxCCS = rindxCCS;
	rindxCCS = cp;
	
	// combine the CRS and CCS indices to form the symmetric pattern
	cp = (int *)malloc( sizeof(int)*nnz*2 );
	rp = (int *)malloc( sizeof(int)*(nrows+1) );
	head = cp;
	rp[0] = 0;
	pos = 0;
	for( i=0; i<nrows; i++ )
	{
		// determine how many indices are in This row
		nr = rindxCRS[i+1] - rindxCRS[i];
		nc = rindxCCS[i+1] - rindxCCS[i];
		n  = nr + nc;
		
		// augment the indices for This row if there are elements to augment 
		if( n )
		{
			// augment the two sets of indices
			if( nr )
				memcpy( head,      cindxCRS  + rindxCRS[i], nr*sizeof(int) );
			if( nc )
				memcpy( head + nr, cindxCCS  + rindxCCS[i], nc*sizeof(int) );
			
			// sort the indices
			heapsort_int( n, head );	
			
			// sort out the unique elements
			j = 0;
			while( j < n-1 )
			{
				cp[pos++] = head[j++];
				if( head[j]==head[j-1] )
					j++;					
			}
			if( j == (n-1) )
			{
				cp[pos++] = head[j];
			}
			
		}
		rp[i+1]=pos;
		head = cp + pos;
	}

	
	// remove extra storage from the end of index_major
	cp = realloc( cp, sizeof(int)*pos );
	free( split[td].index_major);
	free( split[td].index_minor);
	split[td].index_major = cp;
	split[td].index_minor = rp;
	split[td].dim_major = pos;
	
	/*
	 *	recombine the left right and centre
	 */
	// find out how many nonzero are in the symmetric pattern
	n = 0;
	for( i=0; i<nd; i++ )
		n += split[i].dim_major;
	
	// allocate memory for the indices
	index_major = (int *)malloc( sizeof(int)*n );
	index_minor = (int *)malloc( sizeof(int)*(A->mtx.nrows+1) );
	
	// stick 'em together
	pos = 0;
	index_minor[0] = 0;
	for( k=0; k<A->mtx.nrows; k++ )
	{
		for( i=0; i<nd; i++ )
		{
			if( split[i].index_major )
			{
				spot = vtxdist[i];
				for( j=split[i].index_minor[k]; j<split[i].index_minor[k+1]; j++ )
				{
					index_major[pos++] = split[i].index_major[j] + spot;
				}
			}
		}
		index_minor[k+1] = pos;
	}
	
	// now save the information
	*cindx = index_major;
	*rindx = index_minor;
	*length = n;
	
	// free memory
	for( i=0; i<nd; i++ )
		index_free( split + i );
	free( cindxCCS );
	free( rindxCCS );
	
	// if needed, update A by inserting zeros at the new positions
	if( insert )
	{
		int *cpo, *rpo, *cpn, *rpn;
		double *nzp;
		int poso, posn, no, nn;
		
		cpn = index_major;
		rpn = index_minor;
		cpo = A->mtx.cindx;
		rpo = A->mtx.rindx;
		
		if( !A->mtx.block )
		{
			nzp = (double*)calloc( sizeof(double), n );
			
			// loop over the rows
			poso= posn = 0;
			for( i=0; i<nrows; i++ )
			{
				no = rpo[i+1]-rpo[i];
				nn = rpn[i+1]-rpn[i];
				// are the new and old rows different?
				if( no != nn )
				{ 
					// have to copy over by hand, inserting new entries as we find them
					for( k=0; k<nn; )
					{
						// copy the old entry over
						nzp[posn++] = A->mtx.nz[poso++];
						k++;
						
						// skip the zeros
						while( k<nn && cpo[poso]!=cpn[posn] )
						{
							k++; 
							posn++;
						}
					}
				}
				else
				{ 
					// just copy the old row into the new one
					memcpy( nzp + posn, A->mtx.nz + poso, sizeof(double)*no );
					poso += no;
					posn += nn;
				}
			}
		}
		else
		{
			nzp = (double*)calloc( sizeof(double), (n BLOCK_M_SHIFT) );
			posn = poso = 0;
			for( i=0; i<nrows; i++ )
			{
				no = rpo[i+1]-rpo[i];
				nn = rpn[i+1]-rpn[i];
				// are the new and old rows different?
				if( no != nn )
				{ 
					// have to copy over by hand, inserting new entries as we find them
					k=0;
					while( k<nn && cpo[poso]!=cpn[posn] )
					{
						k++; 
						posn++;
					}
					for( ; k<nn; )
					{
						// copy the old entry over
						memcpy( nzp + (posn BLOCK_M_SHIFT), A->mtx.nz + (poso BLOCK_M_SHIFT), BLOCK_SIZE*BLOCK_SIZE*sizeof(double) );
						posn++; poso++;
						k++;
						
						// skip the zeros
						while( k<nn && cpo[poso]!=cpn[posn] )
						{
							k++; 
							posn++;
						}
					}
				}
				else
				{ 
					// just copy the old row into the new one
					memcpy( nzp + (posn BLOCK_M_SHIFT), A->mtx.nz + (poso BLOCK_M_SHIFT), sizeof(double)*(no BLOCK_M_SHIFT) );
					poso += no;
					posn += nn;
				}
			}
			
		}
		// attach and copy the new data to A
		A->mtx.nnz = n;
		free( A->mtx.nz );
		A->mtx.nz = nzp;
		memcpy( rpo, rpn, sizeof(int)*(A->mtx.nrows+1) );
		A->mtx.cindx = (int*)realloc( A->mtx.cindx, sizeof(int)*n );
		memcpy( A->mtx.cindx, cpn, sizeof(int)*(n) );
	}
}

/******************************************************************************************
*	mtx_CRS_dist_gemv()
*	
*   matrix vector product for distributed CRS matrix
*
*   only works for square matrices
*
*   y = A*x
******************************************************************************************/
void mtx_CRS_dist_gemv( Tmtx_CRS_dist_ptr A, Tvec_dist_ptr x, Tvec_dist_ptr y, double alpha, double beta, char trans )
{	
	ASSERT_MSG( A->init, "mtx_CRS_dist_gemv() : matrix must be initialised" );
	ASSERT_MSG( (trans=='N' || trans=='n' || trans=='T' || trans=='t'), "mtx_CRS_dist_gemv() : invalid trans value" );
	/*
	 *  put something here to check that matrix dimensions match
	 */
	
	if( trans=='n' || trans=='N' )
	{
		double *xglobal;
		
		// initialise y
		if( !y->init )
			vec_dist_init( y, &x->This, A->nrows, A->mtx.block, A->vtxdist );
		
		// initialise variables
		if( !A->mtx.block )
			xglobal = (double*)malloc( sizeof(double)*A->nrows );
		else
			xglobal = (double*)malloc( sizeof(double)*(A->nrows BLOCK_V_SHIFT) );
		
		// find the global version of x
		vec_dist_gather_all( x, xglobal );

		// perform my local matrix multiply
		mtx_CRS_gemv( &A->mtx, xglobal, y->dat, alpha, beta, 'N' );
		
		// clean up
		free( xglobal );
	}
	else
	{
		fprintf( stderr, "\n\n\tSTEADY ON, YOU ARE GOING TO HAVE TO IMPLEMENT A DIST TRANSPOSE MULTIPLY FIRST\n\n" );
		/*int i;
		double *ytmp, *yp;
		
		// initialise variables
		if( !A->mtx.block )
			ytmp = (double*)calloc( A->nrows, sizeof(double) );
		else
			ytmp = (double*)calloc( (A->nrows BLOCK_V_SHIFT), sizeof(double) );
		
		// perform my local matrix multiply, store in ytmp
		mtx_CRS_gemv(  &A->mtx, x, ytmp, alpha, 0., 'T' );
		
		// add the ytmps together to get the alpha*A*x part stored on each Pid in ytmp
		vec_MPI_add( ytmp, A->nrows, A->mtx.block, This );
		
		// add This Pid's part of initial beta*y to ytmp, store in y
		if( !A->mtx.block )
		{
			yp = ytmp + A->vtxdist[This->this_proc];
			for( i=0; i<A->mtx.nrows; i++ )
				y[i] = beta*y[i] + yp[i];
		}
		else
		{
			double *yyp;
			
			yp = ytmp + (A->vtxdist[This->this_proc] BLOCK_V_SHIFT);
			for( i=0; i<A->mtx.nrows; i++ )
			{
				yyp = y + (i BLOCK_V_SHIFT);
				BLOCK_V_AXPY( yyp, beta, yyp, yp );
			}
		}
		
		// clean up
		free( ytmp );*/
	}
}

/******************************************************************************************
*	mtx_CRS_dist_residual()
*	
*   calculate the residual for a distributed CRS matrix
*
*   the residual is stored in r, on the root process only. the norm of the residual is
*   also stored on the 
******************************************************************************************/
double mtx_CRS_dist_residual( Tmtx_CRS_dist_ptr A, Tvec_dist_ptr x, Tvec_dist_ptr b, Tvec_dist_ptr r )
{
	double rnorm;
	
	// find -A*x and store in r
	mtx_CRS_dist_gemv( A, x, r, -1., 0., 'N' );
	
	// add b to This to find r
	vec_dist_axpy( b, r, 1. );

	// find the norm of the residual
	rnorm = vec_dist_nrm2( r );
		
	// return the norm of the residual
	return rnorm;
}

/******************************************************************************************
*	domain_copy()
*	
*   copy domain info from one structure to another.
******************************************************************************************/
void domain_copy( Tdomain_ptr from, Tdomain_ptr to )
{
	int n_dom;
	
	to->n_dom = from->n_dom;
	to->n_node = from->n_node;
	n_dom = to->n_dom;
	
	to->vtxdist = (int*)malloc( (n_dom+1)*sizeof(int) );
	memcpy( to->vtxdist, from->vtxdist, (n_dom+1)*sizeof(int) );
	
	to->map = (int*)malloc( (n_dom*n_dom)*sizeof(int) );
	memcpy( to->map, from->map, (n_dom*n_dom)*sizeof(int) );
	
	to->n_in = (int*)malloc( (n_dom)*sizeof(int) );
	memcpy( to->n_in, from->n_in, (n_dom)*sizeof(int) );
	
	to->n_bnd = (int*)malloc( (n_dom)*sizeof(int) );
	memcpy( to->n_bnd, from->n_bnd, (n_dom)*sizeof(int)  );
}

/******************************************************************************************
*	mtx_CRS_dist_gather()
*	
*   gather a Tmtx_CRS_dist matrix A onto the Pid specified by root in the Tmtx_CRS format
*	matrix pointed to by Alocal
******************************************************************************************/
void mtx_CRS_dist_gather( Tmtx_CRS_dist_ptr A, Tmtx_CRS_ptr Alocal, int root )
{
	int nnz, nproc, thisroot=0, i, j;
	int *nnzspace, *counts;
		
	// get us some constants
	nproc = A->This.n_proc;	
	if( root==A->This.this_proc )
		thisroot=1;
	
	// are we working with a distributed matrix distributed over 1 CPU? if so, then the gather is very simple
	if( nproc==1 )
	{
		mtx_CRS_copy( &A->mtx, Alocal );
	}
	else
	{
		
		// determine how many nz
		MPI_Allreduce( &A->mtx.nnz, &nnz, 1, MPI_INT, MPI_SUM, A->This.comm );
		
		if( thisroot )
		{			
			// initialise the local matrix
			mtx_CRS_init( Alocal, A->nrows, A->ncols, nnz, A->mtx.block );
			
			// determine the number of nz on each Pid, and their distribution
			nnzspace = (int*)malloc( sizeof(int)*(nproc+1) );
			counts = (int*)malloc( sizeof(int)*(nproc) );
			MPI_Gather( &A->mtx.nnz, 1, MPI_INT, counts, 1, MPI_INT, root, A->This.comm );
			nnzspace[0] = 0;
			for( i=0; i<nproc; i++ )
				nnzspace[i+1] = nnzspace[i] + counts[i];
			
			// collect the nz values from the slaves
			if( A->mtx.block )
			{
				int *nnzspace_b, *counts_b;
				
				nnzspace_b = (int*)malloc( sizeof(int)*(nproc+1) );
				counts_b = (int*)malloc( sizeof(int)*(nproc) );
				for( i=0; i<nproc; i++ )
				{
					nnzspace_b[i] = nnzspace[i] BLOCK_M_SHIFT;
					counts_b[i] = counts[i] BLOCK_M_SHIFT;
				}
				nnzspace_b[nproc] = nnzspace[nproc] BLOCK_M_SHIFT;
				
				MPI_Gatherv( A->mtx.nz, (A->mtx.nnz BLOCK_M_SHIFT), MPI_DOUBLE, Alocal->nz, counts_b, nnzspace_b, MPI_DOUBLE, root, A->This.comm );
				
				free( nnzspace_b );
				free( counts_b );
			}
			else
			{
				MPI_Gatherv( A->mtx.nz, A->mtx.nnz, MPI_DOUBLE, Alocal->nz, counts, nnzspace, MPI_DOUBLE, root, A->This.comm );
			}
			
			// collect the cindx values from the slaves
			MPI_Gatherv( A->mtx.cindx, A->mtx.nnz,   MPI_INT,    Alocal->cindx, counts,   nnzspace,   MPI_INT,    root, A->This.comm  );
			
			// collect the rindx values from the slaves
			for( i=0; i<nproc; i++ )
				counts[i] = A->vtxdist[i+1] - A->vtxdist[i];			
			MPI_Gatherv( A->mtx.rindx, A->mtx.nrows, MPI_INT,    Alocal->rindx, counts, A->vtxdist, MPI_INT,    root, A->This.comm  );
			
			
			// the rindx values are still in local form, so convert to global form
			for( i=0; i<nproc; i++ )
				for( j=A->vtxdist[i]; j<A->vtxdist[i+1]; j++ )
					Alocal->rindx[j] += nnzspace[i];
			Alocal->rindx[A->nrows] = nnz;
			
			free( nnzspace );
			free( counts );
		}
		else
		{
			// send the number of nnz on this Pid to the root
			MPI_Gather( &A->mtx.nnz, 1, MPI_INT, NULL, 1, MPI_INT, root, A->This.comm );
			
			// send the nz on this Pid to the root
			if( A->mtx.block )
				MPI_Gatherv( A->mtx.nz, (A->mtx.nnz BLOCK_M_SHIFT), MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, root, A->This.comm );
			else
				MPI_Gatherv( A->mtx.nz, A->mtx.nnz, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, root, A->This.comm );
			
			// send the local cindx values to the root Pid
			MPI_Gatherv( A->mtx.cindx, A->mtx.nnz, MPI_INT, NULL, NULL, NULL, MPI_INT, root, A->This.comm  );
			
			// send the local rindx values to the root Pid
			MPI_Gatherv( A->mtx.rindx, A->mtx.nrows, MPI_INT, NULL, NULL, NULL, MPI_INT,  root, A->This.comm  );
		}
	}

}

void mtx_CRS_split_init( Tmtx_CRS_split_ptr A, int n, int n_in, int n_bnd, int n_dom, int this_dom, int n_neigh )
{
	int i;
	
	mtx_CRS_split_free( A );
	
	A->init = 1;
	A->neighbours = (int*)malloc(sizeof(int)*n_neigh);
	A->n_bnd = n_bnd;
	A->n_in = n_in;
	A->n_dom = n_dom;
	A->this_dom = this_dom;
	A->n_neigh = n_neigh;
	A->n = n;
	A->B.init = 0;
	A->C.init = 0;
	A->E.init = 0;
	A->F.init = 0;
	A->Eij = (Tmtx_CRS_ptr)malloc( sizeof(Tmtx_CRS_dist)*n_neigh );
	if( A->Eij )
	{
		for( i=0; i<n_neigh; i++ )
			A->Eij[i].init = 0;
	}
}

void mtx_CRS_split_free( Tmtx_CRS_split_ptr A )
{
	int i;
	
	
	if( !A->init )
		return;
	A->init = 0;
	if( A->neighbours )
		free( A->neighbours );
	mtx_CRS_free( &A->B );
	mtx_CRS_free( &A->C );
	mtx_CRS_free( &A->E );
	mtx_CRS_free( &A->F );
	if( A->Eij )
	{
		for( i=0; i<A->n_neigh; i++ )
			mtx_CRS_free( A->Eij + i );
		free( A->Eij );
	}
}

void profile_output( int *cindx, int *rindx, int nrows, int ncols, int nnz, char *fname )
{
	int i;
	FILE *fid;
	
	fid = fopen( fname, "w" );
	
	fprintf( fid, "%d\n%d\n%d\n", nrows, ncols, nnz );
	for( i=0; i<nrows+1; i++ )
		fprintf( fid, "%d\n", rindx[i] );
	for( i=0; i<nnz; i++ )
		fprintf( fid, "%d\n", cindx[i] );
	fclose( fid );
}
