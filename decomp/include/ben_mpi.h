/*
    ben_mpi.h
    
    contains all of Ben's basic MPI routines, used for communication between
    processes etc.
*/
#ifndef __BEN_MPI_H__
#define __BEN_MPI_H__

#include <mpi.h>

#include "indices.h"

typedef struct MPI_dat
{
    int this_proc;      // the id number of This process
    int n_proc;         // the total number of processes
    MPI_Comm comm;      // communicator of This setup, usually MPI_COMM_WORLD
    struct MPI_dat *sub_comm;
} TMPI_dat, *TMPI_dat_ptr;

extern int  BMPI_init( int argc, char *argv[], TMPI_dat_ptr This );
extern void BMPI_local_comm_create( TMPI_dat_ptr This );
extern void BMPI_copy( TMPI_dat_ptr from, TMPI_dat_ptr to );
extern void BMPI_print( FILE *fid, TMPI_dat_ptr This );
extern void BMPI_free( TMPI_dat_ptr This );

// redundant routines
extern void BMPI_isok( TMPI_dat_ptr This, int ok );
extern int  BMPI_ok( TMPI_dat_ptr This, int root );
extern void BMPI_index_send( Tindex_ptr index, int target_dom, TMPI_dat_ptr This );
extern void BMPI_index_recv( Tindex_ptr index, int source_dom, TMPI_dat_ptr This );


#endif
