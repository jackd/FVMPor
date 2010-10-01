/*
	libalg_dense_mpi.c
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "linalg.h"
#include "ben_mpi.h"
#include "linalg_mpi.h"
#include "linalg_dense_mpi.h"

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_dist_init( Tmtx_dist_ptr A, TMPI_dat_ptr This, int nrows, int ncols, int block, int *vtxdist )
{
	int i;
	
	A->init = 1;
	A->block = block;
	A->mtx.init = 0;
	if( block )
	{
		nrows = nrows BLOCK_V_SHIFT;
		ncols = ncols BLOCK_V_SHIFT;
	}
	mtx_init( &A->mtx, nrows, ncols );
	
	// keep track of who has what
	
	A->vtxdist = (int*)malloc( (This->n_proc+1)*sizeof(int) );
	
	// if vtxdist == NULL then we have to communicate amongst ourselves to determine the vtxdist
	if( !vtxdist )
	{
		MPI_Allgather( &nrows, 1, MPI_INT, A->vtxdist+1, This->n_proc, MPI_INT, This->comm );
		A->vtxdist[0] = 0;
		if( !block )
		{
			for( i=1; i<This->n_proc; i++ )
				A->vtxdist[i+1] += A->vtxdist[i];
		}
		else
		{
			A->vtxdist[1] /= BLOCK_SIZE; 
			for( i=1; i<This->n_proc; i++ )
				A->vtxdist[i+1] = A->vtxdist[i+1]/BLOCK_SIZE + A->vtxdist[i];
		}
	}
	else
	{
		memcpy( A->vtxdist, vtxdist, (This->n_proc+1)*sizeof(int) );
	}
	A->vtxdist_b = (int*)malloc( (This->n_proc+1)*sizeof(int) );
	A->vtxspace = (int*)malloc( (This->n_proc)*sizeof(int) );
	
	memcpy( A->vtxdist_b, vtxdist, (This->n_proc+1)*sizeof(int) );
	if( block )
		for( i=0; i<This->n_proc+1; i++ )
			A->vtxdist_b[i] = A->vtxdist_b[i] BLOCK_V_SHIFT;
	for( i=0; i<This->n_proc; i++ )
		A->vtxspace[i] = A->vtxdist_b[i+1] - A->vtxdist_b[i];	
	
	BMPI_copy( This, &A->This );
}

/******************************************************************************************
*
*	same as for mtx_dist_init but nrows and ncols are stated explitly
*
******************************************************************************************/
void mtx_dist_init_explicit( Tmtx_dist_ptr A, TMPI_dat_ptr This, int nrows, int ncols, int block, int *vtxdist )
{
	int i;
	
	A->init = 1;
	A->block = block;
	A->mtx.init = 0;
	mtx_init( &A->mtx, nrows, ncols );
	
	// keep track of who has what
	
	A->vtxdist = (int*)malloc( (This->n_proc+1)*sizeof(int) );
	
	// if vtxdist == NULL then we have to communicate amongst ourselves to determine the vtxdist
	if( !vtxdist )
	{
		MPI_Allgather( &nrows, 1, MPI_INT, A->vtxdist+1, This->n_proc, MPI_INT, This->comm );
		A->vtxdist[0] = 0;
		if( !block )
		{
			for( i=1; i<This->n_proc; i++ )
				A->vtxdist[i+1] += A->vtxdist[i];
		}
		else
		{
			A->vtxdist[1] /= BLOCK_SIZE; 
			for( i=1; i<This->n_proc; i++ )
				A->vtxdist[i+1] = A->vtxdist[i+1]/BLOCK_SIZE + A->vtxdist[i];
		}
	}
	else
	{
		memcpy( A->vtxdist, vtxdist, (This->n_proc+1)*sizeof(int) );
	}
	A->vtxdist_b = (int*)malloc( (This->n_proc+1)*sizeof(int) );
	A->vtxspace = (int*)malloc( (This->n_proc)*sizeof(int) );
	
	memcpy( A->vtxdist_b, vtxdist, (This->n_proc+1)*sizeof(int) );
	if( block )
		for( i=0; i<This->n_proc+1; i++ )
			A->vtxdist_b[i] = A->vtxdist_b[i] BLOCK_V_SHIFT;
	for( i=0; i<This->n_proc; i++ )
		A->vtxspace[i] = A->vtxdist_b[i+1] - A->vtxdist_b[i];	
	
	BMPI_copy( This, &A->This );
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_dist_free( Tmtx_dist_ptr A )
{
	if( !A->init )
		return;
	
	A->init = 0;
	mtx_free( &A->mtx );
	if( A->vtxdist )
		free( A->vtxdist );
	if( A->vtxdist_b )
		free( A->vtxdist_b );
	if( A->vtxspace )
		free( A->vtxspace );
	A->vtxdist   = NULL;
	A->vtxdist_b = NULL;
	A->vtxspace  = NULL;
	BMPI_free( &A->This );
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_dist_clear( Tmtx_dist_ptr A )
{
	if( !A->init )
		return;
	
	mtx_clear( &A->mtx );
}

/******************************************************************************************
*
*	initialise a distributed vector x. The value of n is the length of the local part of
*	the vector in terms of blocks. If vtxdist is known, then it can be passed to reduce the
*	amount of communication required, otherwise pass NULL and the routine will determine 
*	vtxdist.
*
******************************************************************************************/
void vec_dist_init( Tvec_dist_ptr x, TMPI_dat_ptr This, int n, int block, int *vtxdist )
{
	int i;
	
	vec_dist_free( x );
	
	x->init = 1;
	x->block = block;
	if( !block )
		x->n = n;
	else
		x->n = n BLOCK_V_SHIFT;
	
	x->dat = (double *)calloc( x->n, sizeof(double) );
	x->vtxdist = (int*)malloc( (This->n_proc+1)*sizeof(int) );
	
	// if vtxdist == NULL then we have to communicate amongst ourselves to determine the vtxdist
	if( !vtxdist )
	{
		MPI_Allgather( &n, 1, MPI_INT, x->vtxdist+1, 1, MPI_INT, This->comm );
		x->vtxdist[0] = 0;
		for( i=1; i<This->n_proc; i++ )
			x->vtxdist[i+1] += x->vtxdist[i];
	}
	else
	{
		memcpy( x->vtxdist, vtxdist, (This->n_proc+1)*sizeof(int) );
	}
	
	x->vtxdist_b = (int*)malloc( (This->n_proc+1)*sizeof(int) );
	x->vtxspace = (int*)malloc( (This->n_proc)*sizeof(int) );
	memcpy( x->vtxdist_b, x->vtxdist, (This->n_proc+1)*sizeof(int) );
	if( block )
		for( i=0; i<This->n_proc+1; i++ )
			x->vtxdist_b[i] = x->vtxdist_b[i] BLOCK_V_SHIFT;
	for( i=0; i<This->n_proc; i++ )
		x->vtxspace[i] = x->vtxdist_b[i+1] - x->vtxdist_b[i];
	
	BMPI_copy( This, &x->This );
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void vec_dist_init_vec( Tvec_dist_ptr x, Tvec_dist_ptr y )
{
	// null case where y is not initialised
	ASSERT_MSG( y->init, "vec_dist_init_vec() : y must be initialised" );
	
	vec_dist_free( x );
	
	x->init = 1;
	x->block = y->block;
	x->n = y->n;
	x->dat = (double *)calloc( x->n, sizeof(double) );
	
	x->vtxdist = (int*)malloc( (y->This.n_proc+1)*sizeof(int) );
	x->vtxdist_b = (int*)malloc( (y->This.n_proc+1)*sizeof(int) );
	x->vtxspace = (int*)malloc( (y->This.n_proc)*sizeof(int) );
	memcpy( x->vtxdist, y->vtxdist, (y->This.n_proc+1)*sizeof(int) );
	memcpy( x->vtxdist_b, y->vtxdist_b, (y->This.n_proc+1)*sizeof(int) );
	memcpy( x->vtxspace, y->vtxspace, (y->This.n_proc)*sizeof(int) );
	BMPI_copy( &y->This, &x->This );
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void vec_dist_free( Tvec_dist_ptr x )
{
	if( !x->init )
		return;
	
	x->init = 0;
	
	if( x->vtxdist )
		free( x->vtxdist );
	if( x->vtxdist_b )
		free( x->vtxdist_b );
	if( x->vtxspace )
		free( x->vtxspace );
	if( x->dat )
		free( x->dat );
	x->vtxdist   = NULL;
	x->vtxdist_b = NULL;
	x->vtxspace  = NULL;
	x->dat       = NULL;
	BMPI_free( &x->This );
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void vec_dist_clear( Tvec_dist_ptr x )
{
	int i;
	
	if( !x->init )
		return;
	
	for( i=0; i<x->n; i++ )
		x->dat[i] = 0.;
}

/******************************************************************************************
*	vec_dist_gather()
*	
*   gather a distributed vector to the root process
******************************************************************************************/
void vec_dist_gather( Tvec_dist_ptr xlocal, double *xglobal, int root )
{
	if( xlocal->This.n_proc>1 )
		MPI_Gatherv( xlocal->dat, xlocal->vtxspace[xlocal->This.this_proc], MPI_DOUBLE, xglobal, xlocal->vtxspace, xlocal->vtxdist_b, MPI_DOUBLE, root, xlocal->This.comm );
	else
		memcpy( xglobal, xlocal->dat, sizeof(double)*xlocal->vtxdist_b[xlocal->This.n_proc] );
}

/******************************************************************************************
*   vec_dist_gather_all()
*
*	All Pid get a copy of the vector stored locally on each Pid
*   as xlocal
******************************************************************************************/
void vec_dist_gather_all( Tvec_dist_ptr xlocal, double *xglobal )
{	
	// gather each domain's local ordering for the original node tags
	if( xlocal->This.n_proc>1 )
		MPI_Allgatherv( xlocal->dat, xlocal->vtxspace[xlocal->This.this_proc], MPI_DOUBLE, xglobal, xlocal->vtxspace, xlocal->vtxdist_b, MPI_DOUBLE, xlocal->This.comm );
	else
		memcpy( xglobal, xlocal->dat, sizeof(double)*xlocal->vtxdist_b[xlocal->This.n_proc] );
}

/******************************************************************************************
*	vec_dist_scatter()
*	
*   scatter a locally stored vector as a distributed vector
*
*	n is the length of the local vector
******************************************************************************************/
void vec_dist_scatter( Tvec_dist_ptr x_dist, double *x, TMPI_dat_ptr This, int block, int n, int root )
{			
	// initialise the distributed vector
	vec_dist_init( x_dist, This, n, block, NULL );
		
	// scatter the data
	if( This->n_proc>1 )
		MPI_Scatterv( x,       x_dist->vtxspace, x_dist->vtxdist_b, MPI_DOUBLE, x_dist->dat, x_dist->n, MPI_DOUBLE, root, This->comm );
	else
		memcpy( x_dist->dat, x, sizeof(double)*x_dist->n );
}

void vec_dist_local_scatter( Tvec_dist_ptr x_dist, double *x, TMPI_dat_ptr This, int block, int n )
{			
	// initialise the distributed vector
	vec_dist_init( x_dist, This, n, block, NULL );
	
	// scatter the data
	memcpy( x_dist->dat, x, sizeof(double)*x_dist->n );
}

/******************************************************************************************
*	vec_dist_scatter_vec()
*	
*   The same operation as vec_dist_scatter, however the distrobuted vector is setup
*	to be the same distribution as y, thus the length of x is implicit in the 
*	definition of y. Most often scatter operations are paired with gather operations,
*	so this information is available
******************************************************************************************/
void vec_dist_scatter_vec( Tvec_dist_ptr x, double *x_local, int root, Tvec_dist_ptr y )
{			
	// initialise the distributed vector
	vec_dist_init_vec( x, y );
	
	// scatter the data
	if( y->This.n_proc>1 )
		MPI_Scatterv( x_local, x->vtxspace, x->vtxdist_b, MPI_DOUBLE, x->dat, x->n, MPI_DOUBLE, root, x->This.comm );
	else
		memcpy( x->dat, x_local, sizeof(double)*x->n );
}

/******************************************************************************************
*	mtx_dist_gather()
*	
*   distribute a locally stored matrix section so that the root process has a copy
*   of the global matrix
******************************************************************************************/
void mtx_dist_gather( Tmtx_dist_ptr Alocal, Tmtx_ptr Aglobal, int root )
{
	int  posl, posg, col, ncols, n_proc;

	// setup constants
	ncols = Alocal->mtx.ncols;
	n_proc = Alocal->This.n_proc;
		
	// initialise the target matrix on the root process
	if( Alocal->This.this_proc==root )
	{
		mtx_free( Aglobal );
		mtx_init( Aglobal, Alocal->vtxdist_b[Alocal->This.n_proc], ncols );
	}
	
	// collect all of the matrix values to the root process
	posg = 0;
	posl = 0;
	for( col=0; col<ncols; col++, posg+=Alocal->vtxdist_b[n_proc], posl+=Alocal->mtx.nrows )
		MPI_Gatherv( Alocal->mtx.dat + posl, Alocal->mtx.nrows, MPI_DOUBLE, Aglobal->dat+posg, Alocal->vtxspace, Alocal->vtxdist_b, MPI_DOUBLE, root, Alocal->This.comm );
}

/***************************************************************
*   vec_dist_print()
*
*   print out the complete vector to fid on Pid=root
***************************************************************/
void vec_dist_print( FILE *fid, Tvec_dist_ptr x, int root )
{
	double *dat=NULL;
	int i, dom;
	
	if( !x->init )
	{
		if( root==x->This.this_proc )
			fprintf( fid, "\tEmpty vector\n" );
		return;
	}
	
	if( root==x->This.this_proc )
	{
		dat = (double *)calloc( x->vtxdist_b[x->This.n_proc], sizeof(double) );
	}
	
	vec_dist_gather( x, dat, root );
	
	if( root==x->This.this_proc )
	{
		fprintf( fid, "\n" );
		for( dom=0; dom<x->This.n_proc; dom++ )
		{
			for( i=x->vtxdist_b[dom]; i<x->vtxdist_b[dom+1]; i++ )
			{
				fprintf( fid, "\t(%d\t%d) \t%g\n", dom, i, dat[i] );
			}
		}
		free(dat);
	}
}

/***************************************************************
*   mtx_dist_print()
*
*   print out the complete matrix to fid on Pid=root
***************************************************************/
void mtx_dist_print( FILE *fid, Tmtx_dist_ptr A, int root )
{
	Tmtx Aglobal;
	
	if( !A->init )
	{
		if( root==A->This.this_proc )
			fprintf( fid, "\tEmpty vector\n" );
		return;
	}
	
	mtx_dist_gather( A, &Aglobal, root );
	
	if( root==A->This.this_proc )
	{
		fprintf( fid, "\n" );
		mtx_print( fid, &Aglobal );
		free( &Aglobal );
	}
}

/***************************************************************
****************************************************************
*
*   BLAS level 1 style vector-vector operations
*
****************************************************************
***************************************************************/



/***************************************************************
*   vec_dist_axpy()
*
*	y = alpha*x + y
***************************************************************/
void vec_dist_axpy( Tvec_dist_ptr x, Tvec_dist_ptr y, double alpha )
{
#ifdef DEBUG
	ASSERT_MSG( (x->n == y->n) && (x->init) && (y->init), "vec_dist_axpy() : invalid vectors passed" );
#endif
	daxpy( x->n, alpha, x->dat, 1, y->dat, 1 );
}

/***************************************************************
*   vec_dist_copy()
*
*   y = x
***************************************************************/
void vec_dist_copy( Tvec_dist_ptr x, Tvec_dist_ptr y )
{
	if( !x->init )
	{
		fprintf( stderr, "ERROR : vec_dist_copy() : distributed vector not initialised" );
		MPI_Finalize();
		exit(1);
	}
	
	// initialise y to have the same distribution and length as x
	vec_dist_init_vec( y, x );
	
	// copy over the data
	dcopy( x->n, x->dat, 1, y->dat, 1 );
}

/***************************************************************
*   vec_dist_scale()
*
*  scale a vector by a scalar
*		x = alpha*x
***************************************************************/
void vec_dist_scale( Tvec_dist_ptr x, double alpha )
{
	if( !x->init )
	{
		fprintf( stderr, "ERROR : vec_dist_scale() : distributed vector not initialised" );
		MPI_Finalize();
		exit(1);
	}
	
	dscal( x->n, alpha, x->dat, 1 );
}

/***************************************************************
*   vec_dist_nrm2()
*
*  
***************************************************************/
double vec_dist_nrm2( Tvec_dist_ptr x )
{
	double sum, nrm;
	
	if( !x->init )
	{
		fprintf( stderr, "ERROR : vec_dist_nrm2() : distributed vector not initialised" );
		MPI_Finalize();
		exit(1);
	}
	
	sum = ddot( x->n, x->dat, 1, x->dat, 1 );
	
	MPI_Allreduce( &sum, &nrm, 1, MPI_DOUBLE, MPI_SUM, x->This.comm );
	
	return sqrt(nrm);
}

/***************************************************************
*   vec_dist_normalise()
*
*  
***************************************************************/
double vec_dist_normalise( Tvec_dist_ptr x )
{
	double nrm;
	
	nrm = vec_dist_nrm2( x );
	
	// only divide if nrm is nonzero
	if( nrm )
		dscal( x->n, 1/nrm, x->dat, 1 );
	
	return nrm;
}

/***************************************************************
*   vec_dist_dot()
*
*  
***************************************************************/
double vec_dist_dot( Tvec_dist_ptr x, Tvec_dist_ptr y )
{
	double sum, dot;
	
	
	
	sum = ddot( x->n, x->dat, 1, y->dat, 1 );
	
	MPI_Allreduce( &sum, &dot, 1, MPI_DOUBLE, MPI_SUM, x->This.comm );
	
	return dot;
}

/***************************************************************
*   vec_MPI_add()
*
*  
***************************************************************/
void vec_MPI_add( double *x, int n, TMPI_dat_ptr This )
{
	double *y;
	
	y = (double*)malloc( sizeof(double)*n );
	
	MPI_Allreduce( x, y, n, MPI_DOUBLE, MPI_SUM, This->comm );
	
	memcpy( x, y, sizeof(double)*n );
	free( y );
}



/***************************************************************
****************************************************************
*															   *	
*   BLAS level 2 style matrix-vector operations				   *
*															   *
****************************************************************
***************************************************************/



/***************************************************************
*   mtx_dist_insert_col()
*
*   insert the vector pointed to by x into column col of A
*  
***************************************************************/
void mtx_dist_insert_col( Tmtx_dist_ptr A, int col, Tvec_dist_ptr x )
{	
	ASSERT_MSG( A->init && A->mtx.init && x->init, "mtx_dist_insert_col() : A and x must be initialised" );
	ASSERT_MSG( A->mtx.nrows == x->n, "mtx_dist_insert_col() : A and x have incompatable dimensions" );
	ASSERT_MSG( col>=0 && col<A->mtx.ncols, "mtx_dist_insert_col() : column out of range" );
	
	dcopy( A->mtx.nrows, x->dat, 1, A->mtx.dat + (A->mtx.nrows*col), 1 );
}

/***************************************************************
*   mtx_dist_get_col()
*
*   extract a column from A and store in x
*  
***************************************************************/
void mtx_dist_get_col( Tmtx_dist_ptr A, int col, Tvec_dist_ptr x )
{
	ASSERT_MSG( A->init && A->mtx.init, "mtx_dist_get_col() : A must be initialised" );
	ASSERT_MSG( col>=0 && col<A->mtx.ncols, "mtx_dist_get_col() : column out of range" );
	
	vec_dist_init( x, &A->This, A->vtxdist[A->This.this_proc+1]-A->vtxdist[A->This.this_proc], A->block, A->vtxdist );
	dcopy( A->mtx.nrows, A->mtx.dat + (A->mtx.nrows*col), 1, x->dat, 1 );
}

/***************************************************************
*   mtx_dist_gemv()
*
*   distributed matrix-vector multiplication
*   trans = n or N
*		y = alpha*A*x + beta*y
*   trans = t or T
*		y = alpha*A'*x + beta*y
*   if y is uninitialised 
*   it is initialised to zero, and beta is irrelevant.
*   if y is initialised and of the wrong length an error
*   is thrown
*
*   if we are performing a nontranspose multiply (trans = n or N)
*   there are two ways of passing the vector x. As a distributed vector
*   in x, or as a global copy of the vector pointed to by 
*   yglobal.
*
*   if we are performing a transpose multiply (trans = t or T),
*   there are two ways of returning y. one is as a copy of global
*   y, pointed to by yglobal, or as a distributed vector, stored in
*   y.
***************************************************************/
void mtx_dist_gemv( Tmtx_dist_ptr A, Tvec_dist_ptr x, Tvec_dist_ptr y, double alpha, double beta, int r0, int r1, int c0, int c1, char trans, double *yglobal )
{
	int n, m;
	
	// check for valid parameters
	if( x )
	{	
		ASSERT_MSG( A->init && x->init, "mtx_dist_gemv() : A and x must be initialised" );
	}
	else
	{
		ASSERT_MSG( A->init && yglobal, "mtx_dist_gemv() : A and x must be initialised" );
	}
	ASSERT_MSG( trans=='t' || trans=='T' || trans=='n' || trans=='N', "mtx_dist_gemv() : invalid trans value" );
	ASSERT_MSG( (r0>=0 && r0<=r1 && r1<A->mtx.nrows) && (c0>=0 && c0<=c1 && c1<A->mtx.ncols), "mtx_dist_gemv() : matrix sub-indices out of range"  );
	
	if( trans=='n' || trans=='N' )
	{
		double *xglobal;
		int *vtxdist;
		int i;
		
		
		n = c1-c0+1;
		
		// check that the matrix/vector dimensions match up
		if( x )
		{
			m = x->vtxdist[x->This.n_proc+1];
			ASSERT_MSG( n==m, "mtx_dist_gemv() : A and x dimensions do not match up" );
		}
		if( y->init )
		{
			ASSERT_MSG( y->n==(r1-r0+1), "mtx_dist_gemv() : destination vector y dimensions do not match with those of A" );
		}
		else
		{
			// determine the new vtxdist for the y vector
			vtxdist = (int *)malloc( sizeof(int)*(A->This.n_proc+1) );
			n = r1-r0;
			MPI_Allgather( &n, 1, MPI_INT, vtxdist+1, A->This.n_proc, MPI_INT, A->This.comm );
			vtxdist[0] = 0;
			if( !A->block )
			{
				for( i=1; i<n; i++ )
					vtxdist[i+1] += vtxdist[i];
			}
			else
			{
				vtxdist[1] /= BLOCK_SIZE; 
				for( i=1; i<n; i++ )
					vtxdist[i+1] = vtxdist[i+1]/BLOCK_SIZE + vtxdist[i];
			}
			
			// initialise the y vector
			vec_dist_init( y, &A->This, r1-r0, A->block, vtxdist );
			free( vtxdist );
		}
		
		// get a local copy of the global x vector
		if( !yglobal )
		{
			xglobal = (double*)calloc( x->vtxdist[x->This.n_proc+1], sizeof(double) );
			vec_dist_gather_all( x, xglobal );
		}
		else
		{
			xglobal = yglobal;
		}
		
		// perform my part of the multiplication
		// the result is stored as needed in y
		dgemv( "n", r1-r0+1, c1-c0+1, alpha, A->mtx.dat + (A->mtx.nrows*c0 + r0), A->mtx.nrows, xglobal, 1, beta, y->dat, 1 );
		
		// free up working vector
		if( !yglobal )
		{
			free( xglobal );
		}
	}
	else
	{
		int nc, nr;
		
		nc = c1-c0+1;
		nr = r1-r0+1;
		
		//printf( "nr = %d\tnc = %d\tx->n=%d\n", nr, nc, x->n );
		ASSERT_MSG( nr==x->n, "mtx_dist_gemv() : dimensions of A and x do not agree" );
		ASSERT_MSG( yglobal, "mtx_dist_gemv() : This option not installed yet, result can only be stored in preallocated yglobal vector" );
		
		if( yglobal )
		{
			dgemv( "t", nr, nc, alpha, A->mtx.dat + (A->mtx.nrows*c0 + r0), A->mtx.nrows, x->dat, 1, beta, yglobal, 1 );
			vec_MPI_add( yglobal, nc, &A->This );
		}			
	}
}

