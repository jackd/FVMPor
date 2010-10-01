/*
 *  schur.h
 *  
 *
 *  Created by Ben Cumming on 15/03/06.
 *
 */

#ifndef __SCHUR_H__
#define __SCHUR_H__

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "linalg_mpi.h"
#include "linalg_dense_mpi.h"
#include "ILU.h"
#include "MR.h"
#include "gmres.h"
#include "precon.h"
#include "fileio.h"

typedef struct precon_schur_params
{
	int initial_decomp;
	int nlevels;			// number of levels in preconditioner
	int lfill;
	double droptol;
	int precon_type_B;		// local preconditioner to use on the local Bi
	int precon_type_S;		// type of preconditioner to use on the level 0 Schur compliment
	int GMRES;					// flag whether to use GMRES at the lower level
	char precon_fname_B[256];	// name of file with preconditioner parameters for local Bi
	char precon_fname_S[256];	// name of file with preconditioner parameters for level 0 S
	char GMRES_fname[256];		// name of file with parameters for level 0 GMRES
} Tprecon_schur_params, *Tprecon_schur_params_ptr;

/*
	data type holds the information for one level of a schur compliment preconditioner
 */
typedef struct precon_schur
{
	int init; 
	int root;			// the Pid of the root process to be used
	int initial_decomp; // flags if the matrix that we are preconditioning was already domain decomposed
	Tdomain domains;	// holds global domain information
	Tdistribution forward;
	Tdistribution backward;
	Tgmres GMRES_params;
	int n_in;			// number of internal nodes
	int n_bnd;			// number of boundary nodes
	int n_local;			// number of local nodes = n_in + n_bnd
	int n_neigh;			// number of neighbour domains
	int *part_g; 		// global part map
	int *q;				// global node permutation map (new order of node)
	int *p;				// global node permutation map (old order of node)
	Tmtx_CRS_split A;   // our 4 split from the domain decomposed S from 1 level up
	Tmtx_CRS_dist S;	// our set of rows in the resultant schur compliment matrix
	Tmtx_CRS Slocal;
	int level;
	Tprecon MS;			// the preconditioner for our schur compliment matrix
	Tprecon MB;			// preconditioner for the Bi matrix in A
	TMPI_dat This;		// communicator information for This preconditioner
} Tprecon_schur, *Tprecon_schur_ptr;


extern void precon_schur_init( Tprecon_schur_ptr P, Tmtx_CRS_dist_ptr A  );
extern void precon_schur_free( Tprecon_schur_ptr P );
extern int  precon_schur( Tmtx_CRS_dist_ptr A, Tprecon_ptr P, int level );
extern int  precon_schur_global( Tmtx_CRS_dist_ptr A, Tprecon_ptr P, int level, Tdomain *dom );
extern void precon_schur_apply( Tprecon_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y );
extern int  precon_schur_params_load( Tprecon_schur_params_ptr p, char *fname );
extern void precon_schur_params_copy( Tprecon_schur_params_ptr from, Tprecon_schur_params_ptr to );
extern void precon_schur_params_print( FILE *fid, Tprecon_schur_params_ptr p );
extern void precon_schur_step( Tprecon_schur_ptr P );

extern void schur_form_local( Tprecon_ptr P, Tmtx_CRS_ptr E, Tmtx_CRS_ptr F, Tmtx_CRS_ptr C, Tmtx_CCS_ptr S );

extern void mtx_CRS_dist_split( Tmtx_CRS_dist_ptr A, Tprecon_schur_ptr P );


#endif
