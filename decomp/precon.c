/*
 *	precon.c
 */

#include "precon.h"
#include "schur.h"

int precon_allocate( void **precon, int type );
int precon_isvalidtype( int type );

void precon_print_name( FILE *fid, int type )
{
	switch( type )
	{
		case PRECON_JACOBI :
		{
			fprintf( fid, "PRECON_JACOBI\n" ); break;
		}
		case PRECON_SCHUR	:
		{
			fprintf( fid, "PRECON_SCHUR\n" ); break;
		}
		case PRECON_NONE  :
		{
			fprintf( fid, "PRECON_NONE\n" ); break;
		}
		case PRECON_ILUT :
		{
			fprintf( fid, "PRECON_ILUT\n" ); break;
		}
		case PRECON_ILU0 :
		{
			fprintf( fid, "PRECON_ILU0\n" ); break;
		}
		default :
		{
			fprintf( fid, "PRECON_UNKNOWN\n" ); break;
		}
	}
	
}

void precon_free( Tprecon_ptr precon )
{
	if( !precon->init )
		return;
	
	switch( precon->type )
	{
		case PRECON_NONE  :
		{
			precon->init = 0;
			precon->parameters	= NULL;
			BMPI_free( &precon->This );
			return;
		}
		case PRECON_JACOBI :
		{
			precon_jacobi_free( precon->preconditioner );
			precon_params_free( precon->parameters, precon->type );
			break;
		}
		case PRECON_SCHUR  :
		{
			precon_schur_free( precon->preconditioner );
			precon_params_free( precon->parameters, precon->type );
			
			break;
		}
		case PRECON_ILU0 :
		{
			precon_ILU0_free( precon->preconditioner );
			precon_params_free( precon->parameters, precon->type );
			break;
		}
		default :
		{
			fprintf( stderr, "ERROR : halting, invalid preconditioner type in precon_free()\n\n" );
			MPI_Finalize();
			exit(1);
		}
	}
	free( precon->preconditioner );
	precon->preconditioner	= NULL;
	free( precon->parameters );
	precon->parameters	= NULL;
	
	precon->init = 0;
	BMPI_free( &precon->This );
}

/******************************************************************************************
*  calculate the tag for a communication
*
******************************************************************************************/
int tag_create( int source, int node, int msgtype )
{
	int tag;
	
	tag = msgtype + node*10 + source*1000;
	
	return tag;
}

/******************************************************************************************
*	precon_apply()
*	
*   apply the preconditioner P to the vector x, storing the result in y
*
*   y = P(x)
******************************************************************************************/
void precon_apply( Tprecon_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y )
{	
	// parallel preconditioners
	switch( P->type )
	{
		case PRECON_JACOBI :
		{
			precon_jacobi_apply( (Tprecon_jacobi_ptr)P->preconditioner, x, y );
			break;
		}
		case PRECON_SCHUR :
		{
			precon_schur_apply( P, x, y );
			break;
		}
		case PRECON_NONE :
		{
			vec_dist_copy( x, y );
			break;
		}
		case PRECON_ILUT :
		{
			precon_ILU_apply( (Tprecon_ILU_ptr)P->preconditioner, x, y );
			break;
		}
		case PRECON_ILU0 :
		{
			precon_ILU0_apply( (Tprecon_ILU0_ptr)P->preconditioner, x, y );
			break;
		}
		default :
		{
			fprintf( stderr, "ERROR : invalid preconditioner type (type = %d) passed.", P->type );
			MPI_Abort( P->This.comm, 1 );
			break;
		}
	}
}

int precon( Tprecon_ptr P, Tmtx_CRS_dist_ptr A, void *user_parms )
{
	int success;
	
	if( !P->init )
		return 0;
	
	switch( P->type )
	{
		case PRECON_JACOBI :
		{
			success = precon_jacobi( A, P->preconditioner );
			break;
		}
		case PRECON_SCHUR	:
		{
			Tprecon_schur_params_ptr parms;
			
			parms = P->parameters;
			if( parms->initial_decomp )
				success = precon_schur_global( A, P, parms->nlevels, user_parms );
			else
				success = precon_schur( A, P, parms->nlevels );
			break;
		}
		case PRECON_NONE  :
		{
			success = 1;
			break;
		}
		case PRECON_ILU0 :
		{
			success = precon_ILU0( A, P->preconditioner );
			break;
		}
		default :
		{
			printf( "PRECON_UNKNOWN\n" ); 
			success = 0;
			break;
		}
	}
	
	return success;
}

int precon_params_load( void *params, char *fname, int type )
{	
	
	// load the parameters
	switch( type )
	{
		case PRECON_JACOBI :
		{
			return precon_jacobi_params_load( params, fname );
		}
		case PRECON_SCHUR	:
		{
			return precon_schur_params_load( params, fname );
		}
		case PRECON_NONE  :
		{
			return 1;
		}
		case PRECON_ILU0 :
		{
			return precon_ILU0_params_load( params, fname );
		}
		default :
		{
			fprintf( stderr, "\nERROR : precon_params_load() : invalid preconditioner type\n" ); 
			return 0;
		}
	}	
}

int precon_params_allocate( void **params, int type )
{
	switch( type )
	{
		case PRECON_JACOBI :
		{
			return (((*params)=malloc( sizeof(Tprecon_jacobi_params) ))!=NULL);
		}
		case PRECON_SCHUR	:
		{
			Tprecon_schur_params_ptr p;
			
			if( !(((*params)=malloc( sizeof(Tprecon_schur_params) ))!=NULL) )
				return 0;
			p = *params;
			sprintf( p->precon_fname_B, "%s", "\0" );
			sprintf( p->precon_fname_S, "%s", "\0" );
			sprintf( p->GMRES_fname,    "%s", "\0" );
			return 1;
		}
		case PRECON_NONE  :
		{
			return 1;
		}
		case PRECON_ILU0 :
		{
			return (((*params)=malloc( sizeof(Tprecon_ILU0_params) ))!=NULL);
		}
		default :
		{
			fprintf( stderr, "\nERROR : precon_params_allocate() : invalid preconditioner type\n" ); 
			return 0;
		}
	}
}

int precon_allocate( void **precon, int type )
{
	switch( type )
	{
		case PRECON_JACOBI :
		{
			*precon = calloc( sizeof(Tprecon_jacobi),1 );
			return 1;
		}
		case PRECON_SCHUR	:
		{
			*precon = calloc( sizeof(Tprecon_schur), 1 );
			return 1;
		}
		case PRECON_NONE  :
		{
			*precon = NULL;
			return 1;
		}
		case PRECON_ILU0 :
		{
			*precon = calloc( sizeof(Tprecon_ILU0), 1 );
			return 1;
		}
		default :
		{
			fprintf( stderr, "\nERROR : precon_params_allocate() : invalid preconditioner type\n" ); 
			return 0;
		}
	}
}
	

int precon_load( Tprecon_ptr precon, Tmtx_CRS_dist_ptr A, char *fname )
{
	int type;
	char params_file[256];
	FILE *fid;
	
	// load header data from file
	if( !(fid = fopen(fname,"r")) )
	{
		fprintf( stderr, "ERROR : precon_load() : unable to open precon parameter file %s\n", fname );
		return 0;
	}
	if( !fgetvar( fid, "%d", &type ) )		  return 0;
	if( !fgetvar( fid, "%s", &params_file ) ) return 0;
	
	fclose( fid );
	
	// initialise
	precon_init( precon, A, type );
	if( !precon_params_load( precon->parameters, params_file, type ) )
	{
		fprintf( stderr, "ERROR : precon_load() : unable to open precon parameter file %s\n", params_file );
		return 0;
	}
	
	// need to load extra information for the Schur compliment
	if( precon->type == PRECON_SCHUR )
	{
		Tprecon_schur_ptr s;
		Tprecon_schur_params_ptr p;
		
		s = precon->preconditioner;
		p = precon->parameters;
		
		precon_init( &s->MB, A, p->precon_type_B);
		if( !precon_params_load( s->MB.parameters, p->precon_fname_B, p->precon_type_B ) )	
			return 0;
		
		precon_init( &s->MS, A, p->precon_type_S);
		if( !precon_params_load( s->MS.parameters, p->precon_fname_S, p->precon_type_S ) )	
			return 0;		
	}
	
	return 1;
}

void precon_params_copy( void *from, void *to, int type )
{
	// load the parameters
	switch( type )
	{
		case PRECON_JACOBI :
		{
			//precon_jacobi_params_copy( from, to );
			break;
		}
		case PRECON_SCHUR	:
		{
			precon_schur_params_copy( from, to );		
			break;
		}
		case PRECON_NONE  :
		{
			break;
		}
		case PRECON_ILU0 :
		{
			//precon_ILU0_params_copy( from, to );
			break;
		}
	}
}

void precon_params_free( void *params, int type )
{
	// free the parameters
	switch( type )
	{
		case PRECON_SCHUR	:
		{
			Tprecon_schur_params_ptr P;
			P = params;
			
			sprintf( P->precon_fname_B, "%s", "\0" );
			sprintf( P->precon_fname_S, "%s", "\0" );
			sprintf( P->GMRES_fname, "%s", "\0" );
			break;
		}
	}
}

void precon_params_print( FILE *fid, Tprecon_ptr P )
{
	if( !P->init )
		return;
	
	fprintf( fid, "Preconditioner of type %d : ", P->type );
	precon_print_name( fid, P->type );
	fprintf( fid, "\n\n" );
	
	switch( P->type )
	{
		case PRECON_NONE :
		{
			return;
		}
		case PRECON_ILU0 :
		{
			return;
		}
		case PRECON_SCHUR :
		{
			precon_schur_params_print( fid, P->parameters );
		}
		case PRECON_JACOBI :
		{
			return;
		}
	}	
}

/******************************************************************************************
*	precon_init()
*
*   initialise a preconditioner type
*
*	there are two steps here.
*	The first step is to allocate memory for the preconditioner and its parameters, the
*	second is to initialise the preconditioner and parameter types to default values.
*
*	STEP 1
*	If precon has not been previously initialised, memory is allocated for the
*	preconditioner and and parameters.
*	If precon has been previoulsy allocated to a different type the previous preconditioner
*	and parameters are freed and new memory allocated for the preconditioner and parameters
*
*	STEP 2
*	initialise the preconditioner to default values (and free old preconditioner data if
*   necessary)
*
******************************************************************************************/
void precon_init( Tprecon_ptr precon, Tmtx_CRS_dist_ptr A, int type )
{
	// check that the caller has asked for a valid preconditioner type
	if( !precon_isvalidtype( type ) )
	{
		fprintf( stderr, "ERROR : halting, invalid preconditioner type in precon_init()\n\n" );
		MPI_Finalize();
		exit(1);
	}
	
	// has the preconditioner already been initialised?
	if( !precon->init )
	{
		// this is the first time, so allocate the relevant stuff
		precon_allocate( &precon->preconditioner, type );
		precon_params_allocate( &precon->parameters, type );
		
// BUGFIX 2006
// -----------

// precon needs its This field initialised to "empty" - see below for why

	precon->This.this_proc = 0;
	precon->This.n_proc = 0;
	precon->This.comm = 0;
	precon->This.sub_comm = NULL;

// -----------		
		
		
	}
	// it has previously been allocated to a different type
	else if( precon->type!=type )
	{
		switch( precon->type )
		{			
			case PRECON_NONE  :
			{
				//free( precon->parameters );
				break;
			}
			case PRECON_JACOBI :
			{
				precon_jacobi_free( precon->preconditioner );
				free( precon->preconditioner );
				free( precon->parameters );
				break;
			}
			case PRECON_SCHUR  :
			{
				precon_schur_free( precon->preconditioner );
				free( precon->preconditioner );
				free( precon->parameters );
				break;
			}
			case PRECON_ILU0 :
			{
				precon_ILU0_free( precon->preconditioner);
				free( precon->preconditioner );
				free( precon->parameters );
				break;
			}
				
		}
		precon_allocate( &precon->preconditioner, type );
		precon_params_allocate( &precon->parameters, type );
	}
	
	// initialise the preconditioner
	switch( type )
	{
		case PRECON_NONE  :
		{
			break;
		}
		case PRECON_JACOBI :
		{
			precon_jacobi_init( precon->preconditioner, A );
			break;
		}
		case PRECON_SCHUR  :
		{
			precon_schur_init( precon->preconditioner, A );
			break;
		}
		case PRECON_ILU0 :
		{
			precon_ILU0_init( precon->preconditioner, A );
			break;
		}
	}
	precon->init = 1;
	precon->type = type;

// BUGFIX 2006
// -----------

// The existing MPI data structure must be freed before overwriting it.
// In the case where this is the first time through, the existing structure is
// "empty", such that the free operation is a no-op.

   BMPI_free( &precon->This );

// -----------

	BMPI_copy( &A->This, &precon->This );
}


int precon_isvalidtype( int type ) 
{
	if( type==PRECON_NONE || type==PRECON_ILU0 || type==PRECON_SCHUR || type==PRECON_JACOBI )
		return 1;
	return 0;
}


