#include <stdlib.h>
#include <string.h>

#include "ILU.h"
#include "benlib.h"

char errstring[256];

/**************************************************************
		Performs the ILU(0) factorisation of A inplace
 **************************************************************/
int ILU0_factor( Tmtx_CRS_ptr A, int *diagpos )
{
	int i, j, k=0, kk, jj, kpos, n;
	
	// store the matrix dimension
	n = A->nrows;
	
	ASSERT_MSG( mtx_CRS_validate( A ), "ILU0_factor() : invalid matrix passed in CRS format" );
	

	// find the indices of the diagonal elements
	for( i=0; i<n; i++ )
	{
		k = A->rindx[i];
		while( A->cindx[k]!=i && k<A->rindx[i+1] )
			k++;
		if( k==A->rindx[i+1] )
		{	
			sprintf( errstring, "ILU0_factor() : zero diagonal encountered on row %d/%d\n", i, n );				
			ERROR( errstring );
			return 0;
		}
		diagpos[i] = k;
	}
	
	// scalar entries
	if( !A->block )
	{
		for( i=0; i<n; i++ )
		{
			for( kk = A->rindx[i]; kk<diagpos[i]; kk++ )
			{
				k = A->cindx[kk];
				kpos = A->rindx[k];
				
				A->nz[kk] *= A->nz[diagpos[k]];
				for( jj=kk+1; jj<A->rindx[i+1]; jj++ )
				{
					j = A->cindx[jj];
					
					// locate the element A_kj
					while( A->cindx[kpos]<j && kpos<A->rindx[k+1] )
						kpos++;
					
					// if we have reached the end of row k then there is no need to continue calculating This loop
					if( kpos==A->rindx[k+1] )
						break;
					
					// update the element if needed
					if( A->cindx[kpos]==j )
						A->nz[jj] -= A->nz[kk]*A->nz[kpos]; 
				}
			}
			A->nz[diagpos[k]] = 1./A->nz[diagpos[k]];
		}
	}
	// block entries
	else
	{
		double tmp[ BLOCK_SIZE*BLOCK_SIZE ], *ptr;
		int kkb, jjb, kposb, diagposb;
		
		for( i=0; i<n; i++ )
		{
			for( kk = A->rindx[i], kkb = (A->rindx[i] BLOCK_M_SHIFT); kk<diagpos[i]; kk++, kkb += BLOCK_SIZE*BLOCK_SIZE )
			{				
				k = A->cindx[kk];
				kpos = A->rindx[k];
				kposb = kpos BLOCK_M_SHIFT;
				diagposb = diagpos[k] BLOCK_M_SHIFT;
								
				BLOCK_M_MULT( tmp, A->nz + kkb, A->nz + diagposb );
				BLOCK_M_COPY( tmp, A->nz + kkb );
								
				for( jj=kk+1; jj<A->rindx[i+1]; jj++ )
				{

					j = A->cindx[jj];
					
					// locate the element A_kj
					while( A->cindx[kpos]<j && kpos<A->rindx[k+1] )
						kpos++;
					
					// if we have reached the end of row k then there is no need to continue calculating This loop
					if( kpos==A->rindx[k+1] )
						break;
					
					// update the element if needed
					if( A->cindx[kpos]==j )
					{
						kposb = kpos BLOCK_M_SHIFT;
						jjb = jj BLOCK_M_SHIFT;
						
						BLOCK_M_MULT( tmp, A->nz + kkb, A->nz + kposb );
						BLOCK_M_SUB( A->nz + jjb, A->nz + jjb, tmp );
					}
				}
			}
			
			// invert the factorised diagonal element for row i for easy application
			ptr = A->nz + (diagpos[i] BLOCK_M_SHIFT);
			BLOCK_M_INVERT( ptr );
		}
	}
	
	return 1;
}

int precon_ILU0( Tmtx_CRS_dist_ptr A, Tprecon_ILU0_ptr P )
{
	int success, root=0;
	
	mtx_CRS_dist_gather( A, &P->LU, root );
	
	if( A->This.this_proc==root )
	{
		P->diagpos = malloc( P->LU.nrows*sizeof(int) );
		success = ILU0_factor( &P->LU, P->diagpos );
	}
	MPI_Bcast( &success, 1, MPI_INT, 0, A->This.comm );
	
	return success;
}

void precon_ILU0_apply( Tprecon_ILU0_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y )
{
	int i, j, n, thisroot=0, root=0, local_n;
	int *cindx, *rindx;
	double *nz, *yy;
	
	if( root==x->This.this_proc )
		thisroot = 1;	
	local_n = x->n;
	if( x->block )
		local_n /= BLOCK_SIZE;
	
	// setup the vector for copying
	vec_dist_init_vec( y, x );
	
	if( thisroot )
	{
		// copy x into y and set up a working double pointer
		yy = (double*)malloc( sizeof(double)*x->vtxdist_b[x->This.n_proc] );
		vec_dist_gather( x, yy, root );
		
		// setup some local variables
		n = P->LU.nrows;
		cindx = P->LU.cindx;
		rindx = P->LU.rindx;
		nz    = P->LU.nz;
		
		
		if( !P->LU.block )
		{		
			// Forward subs
			for( i=1; i<n; i++ )
			{
				for( j=rindx[i]; j<P->diagpos[i]; j++ )
				{
					yy[i] -= nz[j]*yy[cindx[j]]; 
				}
			}
			
			// Backward subs
			for( i=n-1; i>=0; i-- )
			{
				for( j=P->diagpos[i]+1; j<rindx[i+1]; j++ )
				{
					yy[i] -= nz[j]*yy[cindx[j]];
				}
				yy[i] *= nz[P->diagpos[i]];
			}
		}
		else
		{
			double *yyp, *nzp, *yyRHSp;
			double tmp[BLOCK_SIZE*BLOCK_SIZE];
			
			// Forward subs
			for( i=1, yyp=yy+BLOCK_SIZE; i<n; i++, yyp+=BLOCK_SIZE )
			{
				nzp = nz + (rindx[i] BLOCK_M_SHIFT);
				for( j=rindx[i]; j<P->diagpos[i]; j++, nzp+=(BLOCK_SIZE*BLOCK_SIZE) )
				{
					yyRHSp = yy + (cindx[j] BLOCK_V_SHIFT);
					BLOCK_MV_SUB_MULT( yyp, nzp, yyRHSp );
				}
			}
			
			// Backward subs
			yyp = yy + ( (n-1) BLOCK_V_SHIFT);
			for( i=n-1; i>=0; i--, yyp-=BLOCK_SIZE  )
			{
				nzp = nz + ((P->diagpos[i]+1) BLOCK_M_SHIFT);
				for( j=P->diagpos[i]+1; j<rindx[i+1]; j++, nzp+=(BLOCK_SIZE*BLOCK_SIZE) )
				{
					yyRHSp = yy + (cindx[j] BLOCK_V_SHIFT);
					BLOCK_MV_SUB_MULT( yyp, nzp, yyRHSp );
				}
				nzp =  nz + (P->diagpos[i] BLOCK_M_SHIFT);
				BLOCK_MV_MULT( tmp, nzp, yyp );
				BLOCK_V_COPY( tmp, yyp );
			}
		}
		
		vec_dist_scatter_vec( y, yy, root, x );		
		free( yy );
	}
	else
	{
		vec_dist_gather( x, yy, root );

		vec_dist_scatter_vec( y, yy, root, x );
	}
}

void precon_ILU_apply( Tprecon_ILU_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y )
{
	vec_dist_copy( x, y );
	
	ILU_fsub( &P->L, y->dat );
	ILU_bsub( &P->U, y->dat );
}

void precon_ILU0_free( Tprecon_ILU0_ptr P )
{

	int Pid;
	
	MPI_Comm_rank( MPI_COMM_WORLD, &Pid );


// BUGFIX 2006
// -----------

// All processes must free their data here.
	
//	if( P->init && !Pid )
	if( P->init )
	{

// BUGFIX 2006
// -----------

// Test (in this case) is harmless, but still unnecessary.  Search for BUGFIX
// to find examples of similar, but harmful tests in this code.

//		if( P->diagpos )
			free( P->diagpos );
// -----------

		P->diagpos = NULL;
		mtx_CRS_free( &P->LU );
	}
	P->init = 0;
}

void precon_ILU0_init( Tprecon_ILU0_ptr P, Tmtx_CRS_dist_ptr A )
{
	precon_ILU0_free( P );
	
	if( A->init )
	{
		P->LU.init = 0;
		P->init = 1;
		P->diagpos = NULL;
	}
}

int precon_ILU0_params_load( Tprecon_ILU0_params_ptr p, char *fname )
{
	// This is just a stub -- there are no specific parameters to load for a Jacobi preconditioner
	FILE * fid;
	
	if( !(fid=fopen(fname,"r")) )
	{
		printf( "precon_ILU0_params_load() : unable to read parameters from file %s\n\n", fname );
		return 0;
	}
	if( !p )
	{
		printf( "precon_ILU0_params_load() : NULL pointer passed\n\n" );
		fclose(fid);
		return 0;
	}
	fclose(fid);
	return 1;
}

/*************************************************
 *
 *
 *		LEGACY
 *
 *
 *************************************************/

void precon_ILU0_apply_serial( Tprecon_ILU0_ptr P, double *x, double *y )
{
	int i, j, n;
	int *cindx, *rindx;
	double *nz, *yy;
	
	// setup some local variables
	n = P->LU.nrows;
	cindx = P->LU.cindx;
	rindx = P->LU.rindx;
	nz    = P->LU.nz;
	
	if( !P->LU.block )
		memcpy( y, x, n*sizeof(double) );
	else
		memcpy( y, x, (n BLOCK_V_SHIFT)*sizeof(double) );
	yy = y;
	
	if( !P->LU.block )
	{	
		// Forward subs
		for( i=1; i<n; i++ )
		{
			for( j=rindx[i]; j<P->diagpos[i]; j++ )
			{
				yy[i] -= nz[j]*yy[cindx[j]]; 
			}
		}
		
		// Backward subs
		for( i=n-1; i>=0; i-- )
		{
			for( j=P->diagpos[i]+1; j<rindx[i+1]; j++ )
			{
				yy[i] -= nz[j]*yy[cindx[j]];
			}
			yy[i] *= nz[P->diagpos[i]];
		}
	}
	else
	{
		double *yyp, *nzp, *yyRHSp;
		double tmp[BLOCK_SIZE*BLOCK_SIZE];
		
		// Forward subs
		for( i=1, yyp=yy+BLOCK_SIZE; i<n; i++, yyp+=BLOCK_SIZE )
		{
			nzp = nz + (rindx[i] BLOCK_M_SHIFT);
			for( j=rindx[i]; j<P->diagpos[i]; j++, nzp+=(BLOCK_SIZE*BLOCK_SIZE) )
			{
				yyRHSp = yy + (cindx[j] BLOCK_V_SHIFT);
				BLOCK_MV_SUB_MULT( yyp, nzp, yyRHSp );
			}
		}
		
		// Backward subs
		yyp = yy + ( (n-1) BLOCK_V_SHIFT);
		for( i=n-1; i>=0; i--, yyp-=BLOCK_SIZE  )
		{
			nzp = nz + ((P->diagpos[i]+1) BLOCK_M_SHIFT);
			for( j=P->diagpos[i]+1; j<rindx[i+1]; j++, nzp+=(BLOCK_SIZE*BLOCK_SIZE) )
			{
				yyRHSp = yy + (cindx[j] BLOCK_V_SHIFT);
				BLOCK_MV_SUB_MULT( yyp, nzp, yyRHSp );
			}
			nzp =  nz + (P->diagpos[i] BLOCK_M_SHIFT);
			BLOCK_MV_MULT( tmp, nzp, yyp );
			BLOCK_V_COPY( tmp, yyp );
		}
		
	}
}

void ILUP( Tmtx_CRS_ptr A, Tprecon_ILU_ptr P, int m, double tau )
{
	int i, j, k, n, pos, Upos, Lpos, Ucol;
	int *index, *ip, *diagpos;
	double *w, wdiag;
	Tmtx_CRS_ptr L, U;
	
	/*
	 *		initialise variables
	 */
	n = A->nrows;
	
	diagpos = (int*)malloc( n*sizeof(int) );
	w = (double*)calloc( n,sizeof(double) );
	index = (int*)malloc( n*sizeof(int) );
	
	L=&P->L;
	U=&P->U;
	U->init = L->init = 0;
	mtx_CRS_init( L, n, n, n*m, A->block );
	mtx_CRS_init( U, n, n, n*m, A->block );	
	L->rindx[0] = U->rindx[0] = 0;
	L->nnz      = U->nnz      = 0;
	Upos		= Lpos		  = 0;
	
	// build a list of the positions of all diagnonal elements, saves lots of work inside the loops
	for( i=0; i<n; i++ )
	{
		diagpos[i] = -1;
		for( j=A->rindx[i]; j<A->rindx[i+1]; j++  )
		{
			if( A->cindx[j]==i )
				diagpos[i] = j;
		}
		
		// check that there is indeed a diagonal entry
		ASSERT_MSG( diagpos[i]!=-1, "ILUP : zero diagonal in matrix A encountered, aborting."  );
	}
	
	for( i=0; i<n; i++ )
	{		
		// set w to equal row i of A
		for( j=A->rindx[i]; j<A->rindx[i+1]; j++  )
			w[A->cindx[j]] = A->nz[j];
		
		// calculate row i of L and U
		for( k = A->cindx[A->rindx[i]]; k<i; k++ )
		{
			if( w[k] )
			{
				// scale diagonal element
				w[k] /= A->nz[diagpos[k]];
				
				// apply first round of dropping - use scaled tolerance
				if( fabs(w[k])<tau )
				{
					w[k] = 0.;
				}
				else
				{
					for( j=U->rindx[k]+1; j<U->rindx[k+1]; j++ )
					{
						Ucol = U->cindx[j];
						w[Ucol] -= w[k]*U->nz[j]; 
					}
				}
			}
		}
		
		/*
		 *		now apply the second round of dropping
		 *
		 *		should make This so that all zeros are stripped off at the start
		 */
		
		// store the diagonal element of w seperately
		wdiag = w[i];
		
		// sort the entries of w by absolute value
		for( j=0; j<n; j++ )
			index[j] = j;
		heapsort_double_index( n, index, w );
		
		ip = index + (n-m);
		
		// sort the largest m values by their column index
		heapsort_int_dindex( m, w + (n-m), ip );
		
		/*
		 *		Three cases :
		 *			(1)	diagpos==-1		no elements at all for the U factor
		 *								just save the diagonal entry to U
		 *			(2) ip[diagpos]==i	diagonal found, and all is good
		 *			(3) diagpos==0		nothing to store in L, make it and eye row
		 *
		 */
		// determine the position of the diagonal element
		pos = -1;
		for( j=0; j<m; j++  )
		{
			// we have found or stepped past the diagnonal element
			if( ip[j]>=i )
			{
				pos = j;
				break;
			}
		}
		
		/*
		 *		store the L and U factors
		 */
		
		// store row i of L
		for( j=0; j<pos; j++, Lpos++ )
		{
			L->nz[Lpos] = w[j+n-m];
			L->cindx[Lpos] = ip[j];
		}
		L->nz[Lpos] = 1.;
		L->cindx[Lpos++] = i;
		L->rindx[i+1] = Lpos;
		
		// store row i of U
		if( ip[pos]!=i )
		{
			U->nz[Upos] = wdiag;
			U->cindx[Upos++] = i;
		}
		for( j=pos; j<m; j++ )
		{
			U->nz[Upos] = w[j+n-m];
			U->cindx[Upos++] = ip[j];
		}
		U->rindx[i+1] = Upos;
		
		// reset
		for( j=0; j<n; j++ )
			w[j] = 0.;
	}
	
	/*
	 *		resize L and U
	 */	
	L->nnz = Lpos;
	L->nz = realloc( L->nz, Lpos*sizeof(double) );
	L->cindx = realloc( L->cindx, Lpos*sizeof(int) );
	U->nnz = Upos;
	U->nz = realloc( U->nz, Upos*sizeof(double) );
	U->cindx = realloc( U->cindx, Upos*sizeof(int) );
	
	/*
	 *		free up memory used by routine
	 */
	free( w );
	free( index );
	free( diagpos );
}


void ILUP_fast( Tmtx_CRS_ptr A, Tprecon_ILU_ptr P, int m, double tau )
{
	int i, j, k, n, pos, Upos, Lpos, Ucol, colmin, colmax, n_band, m_band;
	int *index, *ip, *diagpos;
	double *w, *wp; 
	double wdiag;
	Tmtx_CRS_ptr L, U;
	
	/*
	 *		initialise variables
	 */
	n = A->nrows;
	
	diagpos = (int*)malloc( n*sizeof(int) );
	w = (double*)calloc( n,sizeof(double) );
	index = (int*)malloc( n*sizeof(int) );
	
	L=&P->L;
	U=&P->U;
	U->init = L->init = 0;
	mtx_CRS_init( L, n, n, n*m, A->block );
	mtx_CRS_init( U, n, n, n*m, A->block );	
	L->rindx[0] = U->rindx[0] = 0;
	L->nnz      = U->nnz      = 0;
	Upos		= Lpos		  = 0;
	
	// build a list of the positions of all diagnonal elements, saves lots of work inside the loops
	for( i=0; i<n; i++ )
	{
		diagpos[i] = -1;
		for( j=A->rindx[i]; j<A->rindx[i+1]; j++  )
		{
			if( A->cindx[j]==i )
				diagpos[i] = j;
		}
		
		// check that there is indeed a diagonal entry
		ASSERT_MSG( (diagpos[i]!=-1  && A->nz[diagpos[i]]), "ILUP : zero diagonal in matrix A encountered, aborting."  );
	}
	
	colmax = 0;
	for( i=0; i<n; i++ )
	{
		// determine the bandwidth column bounds for This row
		if( A->cindx[A->rindx[i+1]-1]>colmax )
			colmax = A->cindx[A->rindx[i+1]-1];
		colmin = A->cindx[A->rindx[i]];
		
		// set w to equal row i of A
		for( j=A->rindx[i]; j<A->rindx[i+1]; j++  )
			w[A->cindx[j]] = A->nz[j];
		
		// calculate row i of L and U
		for( k=colmin; k<i; k++ )
		{
			if( w[k] )
			{
				// scale diagonal element
				w[k] /= A->nz[diagpos[k]];
				
				// apply first round of dropping - use scaled tolerance
				if( fabs(w[k])<tau )
				{
					w[k] = 0.;
				}
				else
				{
					for( j=U->rindx[k]+1; j<U->rindx[k+1]; j++ )
					{
						Ucol = U->cindx[j];
						w[Ucol] -= w[k]*U->nz[j]; 
					}
				}
			}
		}
		
		/*
		 *		now apply the second round of dropping
		 *
		 *		should make This so that all zeros are stripped off at the start
		 */
		
		// store the diagonal element of w seperately
		ASSERT_MSG( w[i], "ILUT() : calculated a zero diagonal in factor U." );
		wdiag = w[i];
		
		
		// point to the start of the nonzero band in w
		ip = index + colmin;
		wp = w + colmin;
		n_band = (colmax-colmin+1);
		
		// sort the entries of w by absolute value
		for( j=colmin; j<=colmax; j++ )
			index[j] = j;
		heapsort_double_index( n_band, ip, wp );
		
		// sort the largest min(m,n_band) values by their column index
		if( n_band<m )
			m_band = n_band;
		else
		{
			m_band = m;
			ip = ip + (n_band-m_band); 
			wp = wp + (n_band-m_band);
		}
		
		while( m_band && !(*wp) )
		{
			ip++;
			wp++;
			m_band--;
		}
		
		heapsort_int_dindex( m_band, wp, ip );
		
		
		/*
		 *		Three cases :
		 *			(1)	diagpos==-1		no elements at all for the U factor
		 *								just save the diagonal entry to U
		 *			(2) ip[diagpos]==i	diagonal found, and all is good
		 *			(3) diagpos==0		nothing to store in L, make it and eye row
		 *
		 */
		// determine the position of the diagonal element
		pos = -1;
		for( j=0; j<m_band; j++  )
		{
			// we have found or stepped past the diagnonal element
			if( ip[j]>=i )
			{
				pos = j;
				break;
			}
		}
		
		/*
		 *		store the L and U factors
		 */
		
		// store row i of L
		for( j=0; j<pos; j++, Lpos++ )
		{
			L->nz[Lpos] = wp[j];
			L->cindx[Lpos] = ip[j];
		}
		L->nz[Lpos] = 1.;
		L->cindx[Lpos++] = i;
		L->rindx[i+1] = Lpos;
		
		// store row i of U
		if( ip[pos]!=i )
		{
			U->nz[Upos] = wdiag;
			U->cindx[Upos++] = i;
		}
		for( j=pos; j<m_band; j++ )
		{
			U->nz[Upos] = wp[j];
			U->cindx[Upos++] = ip[j];
		}
		U->rindx[i+1] = Upos;
		
		// reset
		for( j=colmin; j<=colmax; j++ )
			w[j] = 0.;
	}
	
	
	/*
	 *		resize L and U
	 */
	L->nnz = Lpos;
	L->nz = realloc( L->nz, Lpos*sizeof(double) );
	L->cindx = realloc( L->cindx, Lpos*sizeof(int) );
	U->nnz = Upos;
	U->nz = realloc( U->nz, Upos*sizeof(double) );
	U->cindx = realloc( U->cindx, Upos*sizeof(int) );
	
	/*
	 *		free up memory used by routine
	 */
	free( w );
	free( index );
	free( diagpos );
}

/**********************************************************************
*
*		perform forward sub of L factor in an ILU factorisation.
*
*		assumes L has been factored with 1's (Crout) on diagonal.
*
**********************************************************************/
void ILU_fsub( Tmtx_CRS_ptr L, double *x )
{
	int i, j, n;
	
	n = L->nrows;
	
	for( i=1; i<n; i++ )
	{
		for( j=L->rindx[i]; j<L->rindx[i+1]-1; j++ )
		{
			x[i] -= L->nz[j]*x[L->cindx[j]]; 
		}
	}
}

/**********************************************************************
*
*		perform backward sub of U factor in an ILU factorisation.
*
*		assumes Crout factorisation.
*
**********************************************************************/
void ILU_bsub( Tmtx_CRS_ptr U, double *x )
{
	int i, j, n;
	
	n = U->nrows;
	
	for( i=n-1; i>=0; i-- )
	{
		for( j=U->rindx[i]+1; j<U->rindx[i+1]; j++ )
		{
			x[i] -= U->nz[j]*x[U->cindx[j]];
		}
		x[i] /= U->nz[U->rindx[i]];
	}
}


