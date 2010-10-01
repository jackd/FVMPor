/*
 *	precon.h
 */
#ifndef __PRECON_H__
#define __PRECON_H__

// preconditioner type tags
#define PRECON_NONE				0
#define PRECON_SCHUR			1
#define PRECON_JACOBI			2
#define PRECON_MRPC				3
#define PRECON_ILUT				4
#define PRECON_ILU0				5
//#define PRECON_ILUP				6
//#define PRECON_SPAI				7


#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "ben_mpi.h"
#include "benlib.h"
#include "linalg_mpi.h"
#include "linalg_dense_mpi.h"
#include "parmetis.h"
#include "linalg.h"
#include "indices.h"
#include "fileio.h"
#include "MR.h"
#include "ILU.h"

typedef struct precon_param
{
	int type;
	void *params;
} Tprecon_params, *Tprecon_params_ptr;

typedef struct precon
{
	int init;
	int type;
	TMPI_dat This;
	void *preconditioner;
	void *parameters;
	char fname[256];
} Tprecon, *Tprecon_ptr;

/*
 *		function prototypes
 */

// General preconditioner
extern void precon_print_name( FILE *fid, int type );
extern void precon_init( Tprecon_ptr precon, Tmtx_CRS_dist_ptr A, int type );
extern int  precon( Tprecon_ptr precon, Tmtx_CRS_dist_ptr A, void *user_parms );
extern void precon_apply( Tprecon_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y );
extern void precon_getinfo( char *fname, int *type, char *paramfname );
extern void precon_start( Tprecon_ptr P, char *fname, TMPI_dat_ptr This );
extern int precon_params_allocate( void **params, int type );
extern int precon_params_load( void *params, char *fname, int type );
extern int precon_load( Tprecon_ptr precon, Tmtx_CRS_dist_ptr A, char *fname  );
extern size_t precon_param_size( int type );
extern void precon_params_copy( void *from, void *to, int type );
extern void precon_params_free( void *params, int type );
extern void precon_free( Tprecon_ptr precon );
extern void precon_params_print( FILE *fid, Tprecon_ptr P );


#endif

