/******************************************************************************************
*   MR.h
*
*   Minimal Residual based sparse approximate preconditioners
******************************************************************************************/

#include "linalg.h"
#include "linalg_mpi.h"

#ifndef _MR_H_
#define _MR_H_

// Jacbi parameters. 
// empty as there are no parameters for the jacobi at the moment
typedef struct precon_jacobi_params
{
	int null;
} Tprecon_jacobi_params, *Tprecon_jacobi_params_ptr;

typedef struct precon_jacobi
{
	int init;
	Tvec_dist J;
} Tprecon_jacobi, *Tprecon_jacobi_ptr;

extern Tmtx_CCS_ptr MR_CRS( Tmtx_CRS_ptr A, Tmtx_CCS_ptr M0, int ni, int no, int lfill );
extern int precon_jacobi( Tmtx_CRS_dist_ptr A, Tprecon_jacobi_ptr P );
extern void precon_jacobi_apply( Tprecon_jacobi_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y );
extern void precon_jacobi_init( Tprecon_jacobi_ptr P, Tmtx_CRS_dist_ptr A );
extern void precon_jacobi_free( Tprecon_jacobi_ptr P );
extern int precon_jacobi_params_load( Tprecon_jacobi_params_ptr p, char *fname );

#endif
