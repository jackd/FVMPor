/*
	ben_mpi.c

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>
#include <mpi.h>

#include "benlib.h"
#include "ben_mpi.h"
#include "indices.h"

void BMPI_free_recurse( TMPI_dat_ptr This );


/*
	initialise MPI
*/
int BMPI_init( int argc, char *argv[], TMPI_dat_ptr This )
{
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &This->this_proc );
	MPI_Comm_size( MPI_COMM_WORLD, &This->n_proc );
	This->comm = MPI_COMM_WORLD;
	This->sub_comm = NULL;
	return 1;
}

/*
	copy a communicator
*/
void BMPI_copy( TMPI_dat_ptr from, TMPI_dat_ptr to )
{
	to->this_proc = from->this_proc;
	to->comm = from->comm;
	to->n_proc = from->n_proc;
	if( from->sub_comm )
	{
		to->sub_comm = (TMPI_dat_ptr)malloc( sizeof(TMPI_dat) );
		BMPI_copy( from->sub_comm, to->sub_comm );
	}
	else
		to->sub_comm = NULL;
}

void BMPI_print( FILE *fid, TMPI_dat_ptr This )
{
	int level=0, i;
	char lead[20];
	
	while( This!=NULL )
	{
		for( i=0; i<=level; i++ )
			lead[i] = '\t';

                // BUGFIX 2006
                // -----------
                // Beware of "magic numbers" (like the 20 above) all through
                // this code.  I have no idea where these numbers came from, or
                // how they were deemed to be adequate, but in this case at
                // least, where level is initialised to zero, 20 is more than
                // enough.  lead still needs the trailing null though.

                lead[level+1] = '\0';
                // -----------

		fprintf( fid, "%sP%d/%d On communicator %d\n", lead, This->this_proc, This->n_proc-1, This->comm );
		if( This->sub_comm )
		fprintf( fid, "%sSub communicator : \n", lead );
		This = This->sub_comm;
		level++;
	}
}

/*
	broadcast a flag (ok) from This processor to other processors specified by This->comm
*/
void BMPI_isok( TMPI_dat_ptr This, int ok )
{
	MPI_Bcast( &ok, 1, MPI_INT, This->this_proc, This->comm );
}

/*
	receives and returns flag brodcast by root using BMPI_isok()
*/
int BMPI_ok( TMPI_dat_ptr This, int root )
{
	int ok;
	
	MPI_Bcast( &ok, 1, MPI_INT, root, This->comm );
	
	return ok;
}

/****************************************************************************
	BMPI_index_send()

	send an index set to another process
****************************************************************************/
void BMPI_index_send( Tindex_ptr index, int target_dom, TMPI_dat_ptr This )
{
	MPI_Request request;

	// broadcast the size of the major index
	MPI_Isend( &index->dim_major,  1,                MPI_INT, target_dom, 0, This->comm, &request  );
	
	// broadcast the size of the minor index
	MPI_Isend( &index->dim_minor,  1,                MPI_INT, target_dom, 1, This->comm, &request  );
	
	// broadcast the major index
	MPI_Isend( index->index_major, index->dim_major, MPI_INT, target_dom, 2, This->comm, &request  );
	
	// broadcast the minor index
	MPI_Isend( index->index_minor, index->dim_minor+1, MPI_INT, target_dom, 3, This->comm, &request  );
}

/****************************************************************************
BMPI_index_recv()

send an index set to another process
****************************************************************************/
void BMPI_index_recv( Tindex_ptr index, int source_dom, TMPI_dat_ptr This )
{
	int dim_major, dim_minor;
	MPI_Request request;
	
	// broadcast the size of the major index
	MPI_Irecv( &dim_major,  1,                MPI_INT, source_dom, 0, This->comm, &request  );
	
	// broadcast the size of the minor index
	MPI_Irecv( &dim_minor,  1,                MPI_INT, source_dom, 1, This->comm, &request  );
	
	// allocate memory for data
	index_init( index, dim_major, dim_minor );
	
	// broadcast the major index
	MPI_Irecv( index->index_major, index->dim_major, MPI_INT, source_dom, 2, This->comm, &request  );
	
	// broadcast the minor index
	MPI_Irecv( index->index_minor, index->dim_minor+1, MPI_INT, source_dom, 3, This->comm, &request  );
}

/****************************************************************************
BMPI_local_comm_create()

create a communicator of rank one for the calling process, and store in 
This->sub_comm
****************************************************************************/
void BMPI_local_comm_create( TMPI_dat_ptr This )
{	
	MPI_Comm new_comm;

	// allocate memory for the communicator and associated info
	This->sub_comm = (TMPI_dat_ptr)malloc( sizeof(TMPI_dat) );

	// just split and see how it goes
	MPI_Comm_split( This->comm, This->this_proc, This->this_proc, &new_comm );
	This->sub_comm->comm  = new_comm;
	MPI_Comm_rank( This->sub_comm->comm, &This->sub_comm->this_proc );
	MPI_Comm_size( This->sub_comm->comm, &This->sub_comm->n_proc );
	This->sub_comm->sub_comm = NULL;	
}

void BMPI_free( TMPI_dat_ptr This )
{
	if( This->sub_comm )
		BMPI_free_recurse( This->sub_comm );
	This->sub_comm = NULL;
	This->n_proc = This->this_proc = This->comm = 0;
}

void BMPI_free_recurse( TMPI_dat_ptr This )
{
	if( This->sub_comm )
		BMPI_free_recurse( This->sub_comm );
	free( This );
}

