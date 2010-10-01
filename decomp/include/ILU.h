/*
 *  ILU.h
 *  
 *
 *  Created by Ben Cumming on 25/04/05.
 *
 */
#ifndef __ILU_H__
#define __ILU_H__

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <scsl_blas.h>
#include <unistd.h>
#include <math.h>

#include "benlib.h"
#include "ben_mpi.h"
#include "linalg_mpi.h"
#include "fileio.h"
#include "linalg.h"

typedef struct precon_ILU
{
	Tmtx_CRS L;
	Tmtx_CRS U;
} Tprecon_ILU, *Tprecon_ILU_ptr;

typedef struct precon_ILU0
{
	int init;
	Tmtx_CRS LU;
	int *diagpos;
} Tprecon_ILU0, *Tprecon_ILU0_ptr;

// Jacbi parameters. 
// empty as there are no parameters for the jacobi at the moment
typedef struct precon_ILU0_params
{
	int null;
} Tprecon_ILU0_params, *Tprecon_ILU0_params_ptr;

extern void ILUP( Tmtx_CRS_ptr A, Tprecon_ILU_ptr P, int m, double tau );
extern void ILUP_fast( Tmtx_CRS_ptr A, Tprecon_ILU_ptr P, int m, double tau );
extern void ILU_fsub( Tmtx_CRS_ptr L, double *x );
extern void ILU_bsub( Tmtx_CRS_ptr U, double *x );
extern void precon_ILU_apply( Tprecon_ILU_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y );
extern int  ILU0_factor( Tmtx_CRS_ptr A, int *diagpos );
extern int  precon_ILU0( Tmtx_CRS_dist_ptr A, Tprecon_ILU0_ptr P );
extern void precon_ILU0_apply_serial( Tprecon_ILU0_ptr P, double *x, double *y );
extern void precon_ILU0_apply( Tprecon_ILU0_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y );
extern void precon_ILU0_free( Tprecon_ILU0_ptr P );
extern void precon_ILU0_init( Tprecon_ILU0_ptr P, Tmtx_CRS_dist_ptr A );
extern int  precon_ILU0_params_load( Tprecon_ILU0_params_ptr P, char *fname );

#endif
