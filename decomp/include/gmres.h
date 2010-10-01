/******************************************************************************************
*
* gmres.h
*
******************************************************************************************/

#ifndef __GMRES_H__
#define __GMRES_H__

#include "precon.h"

/* a struct used for holding data about a gmres run.
holds info such as subspace size, tolerance etc
so that This data can be passed easily to and from routines. */
typedef struct
{
	int copied;
	int dim_krylov;
	int max_restarts;
	double tol;
	int restarts;
	int nrmcols;  // flag for if columns are to be normalised 
	double *residuals;
	double *errors;
	int *j;
	int *K;
	int k;
	int reorthogonalise;
	double time_precon;
	double time_gmres;
	FILE *diagnostic;
} Tgmres, *Tgmres_ptr;

extern void gmres(   Tmtx_CRS_dist_ptr A, Tvec_dist_ptr b, Tvec_dist_ptr x, Tgmres_ptr run, Tprecon_ptr precon, int root );
extern void gmresFH( Tmtx_CRS_dist_ptr A, Tvec_dist_ptr b, Tvec_dist_ptr x, Tgmres_ptr run, Tprecon_ptr precon, int root );
extern int arnoldi( Tmtx_CRS_dist_ptr A, int m, int K, Tmtx_dist_ptr Q, Tmtx_ptr Hbar, Tvec_dist_ptr w, Tprecon_ptr precon, int reorthogonalise, int root );
extern void GMRES_params_free( Tgmres_ptr parms );
extern int GMRES_params_load( Tgmres_ptr parms, char *fname );
extern void GMRES_params_copy( Tgmres_ptr from, Tgmres_ptr to );

#endif
