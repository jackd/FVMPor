#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>
#include <mpi.h>

#include "linalg.h"
#include "fileio.h"
#include "benlib.h"
#include "linalg_mpi.h"
#include "ben_mpi.h"
#include "parmetis.h"
#include "indices.h"
#include "gmres.h"
#include "precon.h"
#include "schur.h"

void production_schur_test( int argc, char *argv[] );

/*******************************************************************************************
 
*******************************************************************************************/
int main( int argc, char *argv[] )
{
	production_schur_test( argc, argv );
	
	return 0;
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void production_schur_test( int argc, char *argv[] )
{
	//int _debugwait = 1;
	int loop;
	TMPI_dat This;
	Tdomain dom;
	Tmtx_CRS_dist Ad;
	Tvec_dist x, b;
	int i, thisroot, root=0, nloop=7;
	Tgmres run;
	Tprecon P;
	char *fname_stub;
	double precon_time, gmres_time;
	//double *col_norms;

// TESTING 2006
// ------------
//    double* xx = NULL;
// ------------

	Ad.init = Ad.mtx.init  = 0;
	x.init = b.init = 0;
	P.init = 0;
	
	/*
	 *		initialise the MPI stuff
	 */
	
	// setup MPI as per command line
	BMPI_init( argc, argv, &This );
	
	// setup a local communicator for This process
	BMPI_local_comm_create( &This );
		
	if( root==This.this_proc )
		thisroot = 1;
	else
		thisroot = 0;
	
	/*if( thisroot )
		while( _debugwait );
	MPI_Barrier( This.comm );*/

	// Load the matrix
	if( !This.this_proc )
		printf( "loading matrix...\n" );
	fname_stub = argv[1];

	if( !jacobian_load( &Ad, &dom, fname_stub, &This ) )
	{
		if( thisroot )
			printf( "Unable to reload the jacobian data\n\n" );
		MPI_Finalize();
		return;
	}
		
	// create the preconditioner
	if( !This.this_proc )
		printf( "loading preconditioner parameters... %s\n", "./params/precon.txt" );
	
	if( !precon_load( &P, &Ad, "./params/precon.txt") )
	{
		printf( "ERROR : Unable to load preconditioner\n" );
		MPI_Finalize();
		return;		
	}
	if( !This.this_proc )
	{
		printf( "calculating preconditioner... " );
		precon_print_name( stdout, P.type );
	}	
	
	// setup the GMRES parameters
	if( !This.this_proc )
		printf( "loading GMRES info... %s\n", "./params/GMRES.txt" );
	if( !GMRES_params_load( &run, "./params/GMRES.txt" ) )
	{
		printf( "P%d : ERROR loading GMRES info\n", This.this_proc );
		MPI_Finalize();
		exit(1);
	}

	// loop the preconditioner and GMRES calculation to illustrate the solver
	// in a loop
	precon_time = gmres_time = 0.;
	for( loop=0; loop<nloop; loop++ )
	{

		MPI_Barrier( This.comm );
		precon_time = MPI_Wtime() - precon_time;

		// initialise the preconditioner
		// this preserves the parameters in the preconditioner
		precon_init( &P, &Ad, P.type );
		
		if( P.type==PRECON_SCHUR )
		{
			if( !precon( &P, &Ad, &dom ) )
			{
				printf( "ERROR : Unable to form preconditioner\n" );
				MPI_Finalize();
				return;
			}
		}
		else if( !precon( &P, &Ad, NULL ) )
		{
			printf( "ERROR : Unable to form preconditioner\n" );
			MPI_Finalize();
			return;
		}
		
		MPI_Barrier( This.comm );
		precon_time = MPI_Wtime() - precon_time;

		// initialise the vectors
		vec_dist_init( &b, &This, Ad.mtx.nrows, Ad.mtx.block, Ad.vtxdist );
		vec_dist_init_vec( &x, &b );
		
		// make RHS vector by finding b = A*ones
		for( i=0; i<x.n; i++ )
			x.dat[i] = 1;
		
		mtx_CRS_dist_gemv( &Ad, &x, &b, 1., 0., 'N' );

		vec_dist_clear( &x );
		
		// perform GMRES iteration	
//		if( !This.this_proc )
//			printf( "starting GMRES...\n" );
		gmresFH( &Ad, &b, &x, &run, &P, 0 );

		gmres_time += run.time_gmres;
	}
	// print out some stats
	if( !This.this_proc )
		fprintf( stdout, "\nTimes\n\tpreconditioner calculation :\t%g seconds\n\tGMRES iterations :\t\t%g seconds\n", precon_time/(double)nloop, gmres_time/(double)nloop );
	
	MPI_Barrier( This.comm );
	
	if( !This.this_proc )
		printf( "\nFreeing data...\n" );
	
	precon_free( &P );
	vec_dist_free( &x );
	vec_dist_free( &b );
	mtx_CRS_dist_free( &Ad );
	GMRES_params_free( &run );
	BMPI_free( &This );

// BUGFIX 2006
// -----------
        domain_free( &dom );
// -----------
	
	// close MPI
	MPI_Finalize();
}

