/******************************************************************************************
*   MR.c
*
*   Minimal Residual based sparse approximate preconditioners
*   M0 holds the initial estimate and must have enough memory allocated
*   to hold the sparse inverse (lfill*n nz entries)
******************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>
#include <mpi.h>

#include "MR.h"
#include "linalg.h"
#include "linalg_mpi.h"
#include "benlib.h"

Tmtx_CCS_ptr MR_CRS( Tmtx_CRS_ptr A, Tmtx_CCS_ptr M0, int ni, int no, int lfill )
{
	int counti, counto, n, fill, j;
	Tvec_sp mj;
	double *mjd, *rjd, *zd, *wd, *mj_temp;
	int *p;
	double alpha;
	Tmtx_CCS_ptr M, Mtemp;
	
	/*
	 *  INITIALISE VARIABLES
	 */
	
	// initialise constants
	n = A->nrows;
	
	// allocate memory
	M = (Tmtx_CCS_ptr)malloc( sizeof(Tmtx_CCS) );
	mtx_CCS_init( M, M0->nrows, M0->ncols, M0->nnz, M0->block );
	mjd		= (double*)malloc( n*sizeof(double) );
	rjd		= (double*)malloc( n*sizeof(double) );
	zd		= (double*)malloc( n*sizeof(double) );
	wd      = (double*)malloc( n*sizeof(double) );
	mj_temp = (double*)malloc( lfill*sizeof(double) );
	p		=    (int*)malloc( n*sizeof(int) );
	
	/*
	 *  FIND SPARSE APPROXIMATE INVERSE
	 */
	
	// perform outer iterations
	for( counto=0; counto<no; counto++ )
	{		
		printf( "Outer loop %d :\n", counto );
		
		// loop through columns
		for( j=0; j<n; j++ )
		{
			// select the column to work with
			printf( "\tcolumn %d :\n", j );
			mtx_CCS_getcol_sp( M0, &mj, j );
			vec_sp_scatter( &mj, mjd );
			
			// perform inner iterations on column j
			for(counti=0; counti<ni; counti++ )
			{
				printf( "\t\tinner iteration %d\n", counti );
				// find the residual in two steps
				
				// rjd = -A*mjd
				mtx_CRS_gemv( A, mjd, rjd, -1, 0., 'N' );
				
				// rjd = rjd + ej
				rjd[j] += 1.;
				
				// print the residual vector
				printf( "\t\tresidual\n" );
				vec_print( stdout, rjd, n );

				
				// find the norm of the residual and print it
				printf( "\t\t||r|| = %g\n", dnrm2( n, rjd, 1 ) );
				
				//  find z = M*r
				mtx_CCS_mat_vec_mult( M0, rjd, zd, 1 );
				
				//  find w = A*z
				mtx_CRS_gemv( A, zd, wd, 1, 0., 'N' );
	
				//  find alpha
				alpha = ddot( n, rjd, 1, wd, 1 )/ddot( n, wd, 1, wd, 1 );
				
				//	find mj
				daxpy( n, alpha, rjd, 1, mjd, 1 );
			}
			
			// apply dropping to mj
			fill = dense_gather_lfill( mjd, mj_temp, p, n, lfill );

			// add This column to M
			memcpy( M->nz    + (j*lfill), mjd + (n-lfill), fill*sizeof( double ) );
			memcpy( M->cindx + (j*lfill), p   + (n-lfill), fill*sizeof( int ) );
			
			// sort the newly added nz values by row
			heapsort_int_dindex( lfill, M->nz + (j*lfill), M->cindx + (j*lfill) );
		}
		
		// swap around matrix pointers for This outer iteration
		Mtemp = M0;
		M0 = M;
		M  = Mtemp;
	}
	
	/*
	 *  CLEAN UP
	 */
	vec_sp_free( &mj );
	free( mjd );
	free( rjd );
	free( zd );
	free( wd );
	free( p );
	free( mj_temp );
	mtx_CCS_free( M );
	
	/*
	 *  return the preconditioner
	 */
	return M0;
}


/******************************************************************************************
*	calculates the jacobi preconditioner for the sparse matrix A
* 
*	only for non-block applications at the moment, 
* 
*	assumes that J has been initialised properly!
******************************************************************************************/

// parallel version
int precon_jacobi( Tmtx_CRS_dist_ptr A, Tprecon_jacobi_ptr P )
{
	int i, pos, success=1, global_success, row;
	
	// sclar entries
	if( !A->mtx.block )
	{
		// initialise J
		vec_dist_init( &P->J, &A->This, A->mtx.nrows, A->mtx.block, A->vtxdist );
		
		// fill up j
		row = P->J.vtxdist_b[P->J.This.this_proc];
		for( i=0; i<A->mtx.nrows; i++, row++ )
		{
			// search for diagonal element
			pos = A->mtx.rindx[i];
			while( pos<A->mtx.rindx[i+1] && A->mtx.cindx[pos]<row )
				pos++;
			
			// check to see if we have found a zero diagnonal
			if( pos==A->mtx.rindx[i+1] || !A->mtx.nz[pos] )
			{
				success = 0;
				break;
			}
			
			// find the jacobi value for This row
			P->J.dat[i] = 1./A->mtx.nz[pos];
		}
	}
	// block entries
	else
	{
		int bpos;
		
		// initialise J
		vec_dist_init( &P->J, &A->This, A->mtx.nrows, A->mtx.block, A->vtxdist );
		
		// fill up j
		row = P->J.vtxdist[P->J.This.this_proc];
		for( i=0, bpos=0; i<A->mtx.nrows; i++, row++ )
		{
			// search for diagonal element
			pos = A->mtx.rindx[i];
			while( pos<A->mtx.rindx[i+1] && A->mtx.cindx[pos]<row )
				pos++;
			
			// check to see if we have found a zero diagnonal
			if( pos==A->mtx.rindx[i+1] || A->mtx.cindx[pos]!=row )
			{
				success = 0;
				break;
			}
			
			// find the jacobi value for This row
			pos = pos BLOCK_M_SHIFT;
			P->J.dat[bpos++] = 1./A->mtx.nz[pos];
			P->J.dat[bpos++] = 1./A->mtx.nz[pos+(BLOCK_SIZE*BLOCK_SIZE-1)];
		}
		
		if( !success )
		{
			printf( "P%d : error on row %d\n", P->J.This.this_proc, i );
		}
	}
	
	// broadcast if we were successful or not
	MPI_Allreduce( &success, &global_success, 1, MPI_INT, MPI_LAND, A->This.comm );
	
	// return
	return global_success;
}

/******************************************************************************************
*	apply the Jacobi preconditioner J to the vector x, storing the result in y
*
*   y = J(x)
*
*   J, x and y must be initialised and of the same size
******************************************************************************************/

// parallel version
void precon_jacobi_apply( Tprecon_jacobi_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y )
{
	int i;

	vec_dist_init_vec( y, x );

	if(  P->J.n!=x->n )
	printf( "Jacobi : J %d x %d\n", P->J.n, x->n );
	
	ASSERT_MSG( P->J.init && x->init, "precon_jacobi_apply() : parameters not initialiesed" );
	ASSERT_MSG( P->J.n==x->n, "precon_jacobi_apply() : preconditioner dimensions do not match vector dimsensions" );

	
	for( i=0; i<x->n; i++ )
	{
		y->dat[i] = x->dat[i]*P->J.dat[i];
	}
}

void precon_jacobi_free( Tprecon_jacobi_ptr P )
{
	if( P->init )
		vec_dist_free( &P->J );
	P->init = 0;
}

void precon_jacobi_init( Tprecon_jacobi_ptr P, Tmtx_CRS_dist_ptr A )
{
	precon_jacobi_free( P );
	if(  A->init )
	{		
		P->J.init = 0;
		P->init = 1;
	}
}

int precon_jacobi_params_load( Tprecon_jacobi_params_ptr p, char *fname )
{
	// This is just a stub -- there are no specific parameters to load for a Jacobi preconditioner
	FILE * fid;
	
	if( !(fid=fopen(fname,"r")) )
	{
		printf( "precon_jacobi_params_load() : unable to read parameters from file %s\n\n", fname );
		return 0;
	}
	if( !p )
	{
		printf( "precon_jacobi_params_load() : NULL pointer passed\n\n" );
		fclose(fid);
		return 0;
	}
	fclose(fid);
	return 1;
}

