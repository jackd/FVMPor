#ifndef __FILEIO_H__
#define __FILEIO_H__

#include "linalg_sparse.h"
#include "linalg_dense.h"
#include "domain_decomp.h"
#include "linalg_mpi.h"

#define MTXGENERAL   "general"
#define MTXSYMMETRIC "symmetric"
#define MTXGENERALBLOCK "generalblock"

#define MEAN_ROW_LENGTH 20
#define MAX_INPUT_LINE_LENGTH 256

extern int mtx_CCS_load( char *fname, Tmtx_CCS_ptr mtx );
extern int mtx_CCSB_load( char *fname, Tmtx_CCS_ptr mtx );
extern int mtx_CCS_output( Tmtx_CCS_ptr A, char *fname );
extern int mtx_CCS_output_matlab( Tmtx_CCS_ptr A, char *fname );
extern int mtx_CRS_output( Tmtx_CRS_ptr A, char *fname );
extern int mtx_CRS_output_matlab( Tmtx_CRS_ptr A, char *fname );
extern int mesh_load( Tmesh_ptr mesh, char *fname );
extern int mtx_CRS_load( char *fname, Tmtx_CRS_ptr mtx );
extern int mesh_compiled_save( Tmesh_ptr mesh, char *fname );
extern int mesh_compiled_read( Tmesh_ptr mesh, char *fname );
extern int mtx_CRS_dist_load( Tmtx_CRS_dist_ptr A, char *stub, TMPI_dat_ptr This );
extern int mtx_CRS_dist_output( Tmtx_CRS_dist_ptr A, char *stub );
extern int domain_output( Tdomain_ptr dom, char *fname );
extern int domain_load( Tdomain_ptr dom, char *stub, TMPI_dat_ptr This );
extern int jacobian_load( Tmtx_CRS_dist_ptr A, Tdomain_ptr dom, char *stub, TMPI_dat_ptr This );
extern int jacobian_output( Tmtx_CRS_dist_ptr A, Tdomain_ptr dom, char *stub );
extern char *fgetline( FILE *fid, char *line );
extern int fgetvar( FILE *fid, char *format_string, void *var );
extern int domain_print( FILE *fid, Tdomain_ptr dom );

#endif
