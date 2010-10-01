/*
 *  decomp.c
 *  
 *
 *  Created by Ben Cumming on 15/06/05.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>
#include <mpi.h>

#include "linalg_dense.h"
#include "linalg_sparse.h"
#include "linalg.h"
#include "fileio.h"
#include "benlib.h"
#include "linalg_mpi.h"
#include "ben_mpi.h"
#include "parmetis.h"
#include "indices.h"
#include "gmres.h"
#include "MR.h"
#include "precon.h"
#include "ILU.h"
#include "domain_decomp.h"
#include "metis.h"

/******************************************************************************************
*
******************************************************************************************/
void main( int argc, char *argv[] )
{
	TMPI_dat This;
	Tmesh mesh;
	double time;
	int root=0;
	Tmtx_CRS_dist A;
	Tprecon_schur P;
	char fname[100];
	int thisroot;
	
	// begin MPI
	BMPI_init( argc, argv, &This );
	if( root==This.this_proc )
		printf( "\n----------------------------------------------------------\nDomain decomposition for the file %s.\n----------------------------------------------------------\n\n", fname );
	
	if( root==This.this_proc )
		thisroot=1;
	else
		thisroot=0;
	
	if( argc!=3 )
	{
		if( thisroot )
		{
			printf( "\n\nERROR : wrong number of input arguments.\nmpirun -np N nusage >>./decomp inputmesh.dat outputstub\n" );
			printf( "for example : >>mpirun -np 16 ./decomp mesh.dat mesh_out\nreads mesh data from mesh.dat and outputs the compiled mesh in mesh_out.mesh\nand the 16 domain permutation in mesh_out_16.perm\n" );
		}
		MPI_Finalize();
		exit(1);
	}
	
	// make the root process load up the mesh
	if( thisroot )
	{	
		printf( "Loading mesh...\n" );
		
		// load the mesh, timing how long it takes
		time = get_time();
		mesh_load( &mesh, argv[1] );
		time = get_time() - time;
		
		// print out some diagnostics, including the jacobian sparsity
		printf( "\tmesh loaded in %g seconds. %d nodes and %d elements\n\n", time, mesh.n_nodes, mesh.n_elements );
		mtx_CRS_output( &mesh.A, "Jpattern.mtx" );
	}
	
	
	// everyone waits here for the root process to load the domain
	MPI_Barrier( This.comm );
	
	/*
	 *		split the mesh amongst the domains
	 */
	
	if( thisroot )
	{
		printf( "Splitting mesh...\n" );
		time = get_time();
		
		mtx_CRS_distribute( &mesh.A, &A, &This, root );
		precon_schur_init( &P, &A  );
		mtx_CRS_dist_domdec( &A, &P.forward, P.part_g );
		
		time = get_time() - time;
		printf( "\tmesh split in %g seconds.\n", time );
	}
	else
	{
		mtx_CRS_distribute( NULL, &A, &This, root );
		precon_schur_init( &P, &A  );
		mtx_CRS_dist_domdec( &A, &P.forward, P.part_g );
	}
	
	/*
	 *		output the compiled node/element data and domain decomp info
	 */
	
	if( thisroot )
	{
		FILE *fid;
		
		// only the root Pid has to ouput the compiled node data
		sprintf( fname, "%s.mesh", argv[2] );
		if( !mesh_compiled_save( &mesh, fname ) )
			printf( "ERROR : unable to save the global compiled mesh\n\n" );
		
		
		// output the domain decomp data
		sprintf( fname, "%s_%d.perm", argv[2], This.n_proc );
		fid = fopen( fname, "wb" );
		fwrite( &This.n_proc, sizeof(int), 1, fid );
		printf( "n_proc = %d\n", This.n_proc );
		fwrite( A.vtxdist, sizeof(int), This.n_proc+1, fid );
		fwrite( P.part_g, sizeof(int), A.nrows, fid );
		/*fwrite( P.forward.n_dom, sizeof(int), 1, fid );
		fwrite( P.forward.n_node, sizeof(int), 1, fid );
		fwrite( P.forward.n_neigh, sizeof(int), 1, fid );
		fwrite( P.forward.neighbours, sizeof(int), P.forward.n_neigh, fid );
		fwrite( P.forward.counts, sizeof(int), P.forward.n_dom, fid );
		fwrite( P.forward.index, sizeof(int), , fid );
		fwrite( P.forward.part, sizeof(int), , fid );
		fwrite( P.forward.ppart, sizeof(int), , fid );*/
		/*
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
		*/
		fclose( fid );
	}
	
	// everyone waits for the master, it is just good manners.
	MPI_Barrier( This.comm );
	
	// free up memory
	mtx_CRS_dist_free( &A );
	precon_schur_free( &P );
	
	// finish up
	MPI_Finalize();
}

