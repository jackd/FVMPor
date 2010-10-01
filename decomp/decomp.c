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
#include "schur.h"

/******************************************************************************************
*
******************************************************************************************/
int main( int argc, char *argv[] )
{
    TMPI_dat This;
    Tmesh mesh;
    double time;
    int root=0;
    Tmtx_CRS_dist A;
    Tprecon_schur P;
    char fname[100];
    int thisroot=0, success=0;

    A.init = P.init = 0;

    // begin MPI
    BMPI_init( argc, argv, &This );
    sprintf( fname, "%s.mesh", argv[1] );
    if( root==This.this_proc )
        printf( "\n----------------------------------------------------------\n" );
        printf( "Domain decomposition for the file %s\n", fname );
        printf( "-----------------------------------------------------------\n\n" );

    if( root==This.this_proc )
        thisroot=1;
    else
        thisroot=0;

    if( argc!=2 )
    {
        if( thisroot )
        {
            printf( "\n\nERROR : wrong number of input arguments.\nmpirun -np N nusage >>./decomp inputmesh \n" );
            printf( "for example : >>mpirun -np 16 ./decomp mesh\nreads mesh data from mesh.mesh and outputs the compiled mesh in mesh_16.bmesh\nand the 16 domain permutation in mesh_16.perm\n" );
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

        if( !(success=mesh_load( &mesh, fname )) )
        {
            MPI_Bcast( &success, 1, MPI_INT, root, This.comm );
            MPI_Finalize();
            exit(1);
        }
        MPI_Bcast( &success, 1, MPI_INT, root, This.comm );
        time = get_time() - time;

        // print out some diagnostics, including the jacobian sparsity
        printf( "\tmesh loaded in %g seconds. %d nodes and %d elements\n\n", time, mesh.n_nodes, mesh.n_elements );
        mtx_CRS_output( &mesh.A, "Jpattern.mtx" );
    }
    else
    {
        MPI_Bcast( &success, 1, MPI_INT, root, This.comm );
        if( !success )
        {
            MPI_Finalize();
            exit(1);
        }
    }


    // everyone waits here for the root process to load the domain
    MPI_Barrier( This.comm );

    /*
     *      split the mesh amongst the domains
     */

    if( thisroot )
    {
        printf( "Splitting mesh...\n" );
        time = get_time();
        printf( "\tdistributing from root to slaves...\n" );
        mtx_CRS_distribute( &mesh.A, &A, &This, root );
        precon_schur_init( &P, &A  );
        printf( "\tperforming domain decomposition with ParMETIS...\n" );
        mtx_CRS_dist_domdec_sym( &A, &P.forward, P.part_g );

        time = get_time() - time;
        printf( "\tmesh split in %g seconds.\n", time );
    }
    else
    {
        mtx_CRS_distribute( NULL, &A, &This, root );
        precon_schur_init( &P, &A  );
        mtx_CRS_dist_domdec_sym( &A, &P.forward, P.part_g );
    }

    /*
     *      output the compiled node/element data and domain decomp info
     */

    if( thisroot )
    {
        FILE *fid;

        // only the root Pid has to ouput the compiled node data
        sprintf( fname, "%s_%d.bmesh", argv[1], This.n_proc );
        if( !mesh_compiled_save( &mesh, fname ) )
            printf( "ERROR : unable to save the global compiled mesh\n\n" );


        // output the domain decomp data
        sprintf( fname, "%s_%d.perm", argv[1], This.n_proc );
        fid = fopen( fname, "wb" );
        fwrite( &This.n_proc, sizeof(int), 1, fid );
        fwrite( A.vtxdist, sizeof(int), This.n_proc+1, fid );
        fwrite( P.part_g, sizeof(int), A.nrows, fid );

        fclose( fid );
    }

    // everyone waits for the master, it is just good manners.
    MPI_Barrier( This.comm );

    // free up memory
    mtx_CRS_dist_free( &A );
    precon_schur_free( &P );

    // finish up
    MPI_Finalize();
    return 1;
}

