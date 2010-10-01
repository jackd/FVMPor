#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "benlib.h"
#include "linalg_sparse.h"
#include "linalg_dense.h"

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_init( Tmtx_ptr A, int nrows, int ncols )
{
	if( A->init )
		mtx_free( A );
	
	A->init = 1;
	A->nrows = nrows;
	A->ncols = ncols;
	A->dat = (double *)calloc( nrows*ncols, sizeof(double) );
	A->tau = NULL;		// This is only allocated by least squares routines
	ASSERT_MSG( A->dat!=NULL, "mtx_init() : unable calloc memory for dense matrix" );
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_free( Tmtx_ptr A )
{
	if( !A->init )
		return;
	
	A->nrows = 0;
	A->ncols = 0;
	A->init = 0;
	if( A->dat )
		free( A->dat );
	if( A->tau )
		free( A->tau );
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_clear( Tmtx_ptr A )
{
	int i, m;
	
	if( !A->init )
		return;
	
	m = A->nrows*A->ncols;
	
	for( i=0; i<m; i++ )
		A->dat[i] = 0.;
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_copy( Tmtx_ptr from, Tmtx_ptr to )
{
#ifdef DEBUG
	// check that the matrix we are copying from is initialised
	ASSERT_MSG( from->init, "mtx_copy() : attempt to copy from uninitialised matrix" );
#endif
	
	// initialise the target matrix
	if( to->init )
		mtx_free( to );
	mtx_init( to, from->nrows, from->ncols );
	
	// copy the data
	dcopy( from->nrows*from->ncols, from->dat, 1, to->dat, 1 );
	if( from->tau )
	{
		if( to->nrows>to->ncols )
		{
			to->tau = (double*)malloc( to->nrows*sizeof(double) );
			dcopy( from->nrows, from->tau, 1, to->tau, 1 );
		}
		else
		{
			to->tau = (double*)malloc( to->ncols*sizeof(double) );
			dcopy( from->ncols, from->tau, 1, to->tau, 1 );
		}
	}
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_print( FILE *stream, Tmtx_ptr A )
{
	int row, col;
	
#ifdef DEBUG
	ASSERT_MSG( A->init, "mtx_print() : attempting to print an uninitialised matrix" );
#endif
	
	fprintf( stream, "the matrix is %dX%d\n", A->nrows, A->ncols );
	for( row=0; row<A->nrows; row++ )
	{
		for( col=0; col<A->ncols; col++ )
			fprintf( stream,  "%13.4g", A->dat[row+col*A->nrows] );
		fprintf( stream, "\n" );
	}
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void mtx_givens( Tmtx_ptr A, int k, double *rhs )
{
	int i, pos, lwork, one=1, lda;
	int info;
	double a, b, c, s;  
	double *work, *tau;
	char L = 'L', T = 'T'; 
	
	lda = A->nrows;
	lwork = 64*(k+1);
	work = (double*)calloc( lwork, sizeof(double) );
	tau  = (double*)calloc( k+10, sizeof(double) );
	

	//mtx_QR( A );
	//dormqr_( &L, &T, &A->nrows, &one, &A->ncols, A->dat, &lda, A->tau, rhs, &lda, work, &lwork, &info );
	//return;

	
	/* 
	 *  eliminate upper block if it exists, use Householder reflections 
	 */
	if( k )
	{
		int kplus1 = k+1;
		
		pos = A->ncols-k;
		
		// find QR decomp of block, storing in the block 
		dgeqrf_( &kplus1, &k, A->dat, &lda, tau, work, &lwork, &info );
		// adjust the other columns in the first k+1 rows
		dormqr_( &L, &T, &kplus1, &pos, &k, A->dat, &lda, tau, A->dat+lda*k, &lda, work, &lwork, &info );
		// adjust the first k+1 elements of b
		dormqr_( &L, &T, &kplus1, &one, &k, A->dat, &lda, tau, rhs, &lda, work, &lwork, &info );
	}
	
	/*  eliminate sub-diagonal elements in remaining columns using Given's rotations */
	for( i=k, pos=k*lda; i<A->ncols; i++, pos+=lda )
	{
		a = A->dat[pos+i];
		b = A->dat[pos+i+1];
		drotg( &a, &b, &c, &s );
		drot( A->ncols-i, &A->dat[pos+i], lda, &A->dat[pos+i+1], lda, c, s );
		drot( 1, rhs+i, lda, rhs+i+1, lda, c, s );
	}
	
	free( tau );
	free( work );
}

/******************************************************************************************
*
*	perform QR decomposition of A using LAPACK, storing Q and R in A
*
******************************************************************************************/
void mtx_QR( Tmtx_ptr A )
{
	int lwork, info; 
	double *work;
	
	lwork = 64*(A->ncols+1);
	work = (double*)calloc( lwork, sizeof(double) );
	if( A->tau )
		free( A->tau );
	if( A->nrows>A->ncols )
		A->tau  = (double*)calloc( A->nrows, sizeof(double) );
	else
		A->tau  = (double*)calloc( A->ncols, sizeof(double) );
	
	// find QR decomp of block, storing in the block 
	dgeqrf_( &A->nrows, &A->ncols, A->dat, &A->nrows, A->tau, work, &lwork, &info );
	
	free( work );
}

/******************************************************************************************
*
*	Extract Q factor of stored in terms of elementary reflectors in A, uses LAPACK
*
******************************************************************************************/
void mtx_QR_extractQ( Tmtx_ptr A )
{
	int lwork, info; 
	double *work;
	
	lwork = 64*(A->ncols+1);
	work = (double*)calloc( lwork, sizeof(double) );
	
	// find QR decomp of block, storing in the block 
	dorgqr_( &A->nrows, &A->ncols, &A->ncols, A->dat, &A->nrows, A->tau, work, &lwork, &info );
	
	free( work );
}

/******************************************************************************************
*
*
*
******************************************************************************************/
void vec_print( FILE *stream, double *x, int n )
{
	int i;
	
	if( !x )
		return;
	
	for( i=0; i<n; i++ )
	{
		fprintf( stream, "\t(%d)\t%g\n", i, x[i] );
	}
}

/******************************************************************************************
 *		mtx_eigs( Tmtx_ptr A, double *lambdar, double *lambdai, double *v )
 * 
 *		PRE : A initialised with data, real matrix square
 *		lda is leading dim of A in memory
 *		lambdai, lambdar initialised length 2*n
 *		v initialised length (n^2)
 ******************************************************************************************/
int mtx_eigs( Tmtx_ptr A, double *lambdar, double *lambdai, double *v )
{
	char N='N', jobvr='V';
	int lda=A->ncols, ldvl=1, ldvr, lwork, ilo, ihi, info;
	int *nnull;
	double *work, *null, *scale, *rconde, *rcondv, abnrm;
	
	lwork = 65*A->nrows;
	ldvr  = A->nrows;
	
	null=NULL;
	nnull=NULL;
	work   = (double*)malloc( lwork*sizeof(double) );
	scale  = (double*)malloc( A->nrows*sizeof(double) );
	rconde = (double*)malloc( A->nrows*sizeof(double) );
	rcondv = (double*)malloc( A->nrows*sizeof(double) );
		
	/*dgeevx( N, N, jobvr, N, A->nrows, A->dat, lda, 
				 lambdar, lambdai, null, ldvl, v, ldvr, 
				 &ilo, &ihi, scale, &abnrm, rconde, rcondv, 
				 work, lwork, nnull, &info );*/
	
	dgeevx_( &N, &N, &jobvr, &N, &A->nrows, A->dat, &lda, 
			 lambdar, lambdai, null, &ldvl, v, &ldvr, 
			 &ilo, &ihi, scale, &abnrm, rconde, rcondv, 
			 work, &lwork, nnull, &info );
	
	free( work );
	free( scale );
	free( rconde );
	free( rcondv );
	
	if( info<0 )
	{
		printf( "ERROR : eigenvalue problem, argument %d was bad\n", -info );
		return 0;
	}
	else if( info>0 )
	{
		printf( "ERROR : eigenvalue problem, unable to find all eigenvaluues/vectors\n" );
		return 0;
	}
	return 1;
}

/******************************************************************************************
*	int vec_drop( double* v, int *indx, int n, int lfill, int diag, double tol )
*
*	routine to apply dropping to a dense vector v, of length n. 
*
*	The lfill largest elements in absolute value, and larger than tol, are returned in 
*	the start of v, with their indices in the original array returned in indx. 
*
*	The returned integer is the number of elements in the dropped array. 
*
*	if 0<diag<n, then the value in v[n] is returned in the drop array. 
*
******************************************************************************************/
int vec_drop( double* v, int *indx, int n, int lfill, int diag, double tol )
{
	int pos, i, m, dpos=-1, fill;
	double val, mmax=0;
	
	// find all of the entries that are larger than the tol
	for( i=0, m=0; i<n; i++ )
	{
		val = fabs(v[i]);
		if(  val>tol || diag==i )
		{
			indx[m] = i;
			v[m++] = val;
		}
		if( val>mmax )
			mmax = val;
		if( diag==i )
			dpos = m-1;
	} 
	
	// look after the diag element
	if( dpos>=0 )
		v[dpos] = mmax+1.;
	
	// sort the entries, if there are more than lfill of them
	if( m>lfill )
	{
		// sort the entries of u and permute indx at the same time
		heapsort_double_index( m, indx, v );
		// sort the lfill largest entries of indx
		heapsort_int_dindex( lfill, v + m - lfill, indx + m - lfill );
	}
	
	// pack the entries into the start of v
	pos = m-lfill;
	fill = (m<lfill) ? m : lfill;
	for( i=0; i<fill; i++, pos++ )
	{
		v[i] = v[pos];
		indx[i] = indx[pos];
		if( indx[i]==diag )
			v[i]-=1.;
	}
	
	// return
	return fill;
}

/******************************************************************************************
*	int vec_drop_block( double* v, int *indx, int n, int lfill, int diag, double tol )
*
*	the same as vec_drop(), however for blocks of dimension BLOCK_SIZExBLOCK_SIZE. see 
*	documentation of vec_drop() for more information.
*
*	additionally requires a  vector u, of length n that is used as workspace
*
******************************************************************************************/
int vec_drop_block( double* v, int *indx, int n, int lfill, int diag, double tol, double *u )
{
	int pos, ipos, i, j, k, m, p, dpos=-1, shift, fill, from, to;
	double val, mmax=0;
	
	shift = (n BLOCK_V_SHIFT)-BLOCK_SIZE;
	
	// find all of the entries that are larger than the tol and pack them into u, and their indices
	// into indx
	for( i=0, pos=0, m=0; i<n; i++, pos+=BLOCK_SIZE )
	{
		// this test probably has to change, it is hardwired for BLOCK_SIZE=2
		ipos = pos;
		val = 0.;
		for( j=0; j<BLOCK_SIZE*BLOCK_SIZE; ipos++ )
		{
			val += fabs(v[ipos]);
			if( !((++j)%BLOCK_SIZE) )
				ipos+=shift;
		}
		if(  val>tol || diag==i )
		{
			indx[m] = i;
			u[m++] = val;
		}
		if( val>mmax )
			mmax = val;
		if( diag==i )
			dpos = m-1;
	} 
	
	// look after the diag element
	if( dpos>=0 )
		u[dpos] = mmax+1.;

	// sort the entries, if there are more than L of them
	if(  m>lfill )
	{
		// sort the entries of u and permute indx at the same time
		heapsort_double_index( m, indx, u );
		// sort the L largest entries of indx
		heapsort_int( lfill, indx + m - lfill );
	}


	// pack the entries into the start of v
	fill = (m<lfill) ? m : lfill; 
	from = 0;
	to = 0;
	for( p=0; p<BLOCK_SIZE; p++ )
	{
		to   = p*(fill BLOCK_V_SHIFT);
 		j    = (m<lfill) ? 0 : m - lfill;
		for( i=0; i<fill; i++, j++)
		{
			pos = (indx[j] BLOCK_V_SHIFT) + from;
			for( k=0; k<BLOCK_SIZE; k++, to++, pos++ )
				v[to] = v[pos];
		}
		from += (n BLOCK_V_SHIFT);
	}
	j = (m<lfill) ? 0 : m - lfill;
	for( i=0; i<fill; i++, j++ )
		indx[i] = indx[j];
	
	// return
	return fill;
}  

//  length(u) and length(indx) >= lfill+1
int vec_drop_block_( double* v, int *indx, int n, int lfill, int diag, double tol, double *u )
{
	int pos, ipos, i, j, k, m, p, shift, fill, from, to;
	double val, drop_min;
	
	shift = (n BLOCK_V_SHIFT)-BLOCK_SIZE;
	
	// search the vector v and look find the indices of the lfill entries larger than tol
	drop_min = 0.;
	//printf( "starting sort\n" );
	for( i=0, pos=0, m=0; i<n; i++, pos+=BLOCK_SIZE )
	{
		// find the size of element i
		ipos = pos;
		val = 0.;
		for( j=0; j<BLOCK_SIZE*BLOCK_SIZE; ipos++ )
		{
			val += fabs(v[ipos]);
			if( !((++j)%BLOCK_SIZE) )
				ipos+=shift;
		}
		// is the entry larger than the minimum in the dropped list so far?
		if( val>tol && (val>drop_min || m<lfill) )
		{
			p = binary_search_double_bracket(m, u, val);
			//printf( "\t\tfound %g at %d and put in place %d with searchlen %d and drop_min=%g and lfill-1=%d\n", val, i, p, m, drop_min, lfill-1 );
			
			for( k=m; k>p; k-- )
			{
				indx[k+1]=indx[k];
				u[k+1]=u[k];
			}
			u[p+1] = val;
			indx[p+1] = i;
			m = (m<lfill) ? m+1 : lfill;
			drop_min = (m<lfill) ? 0 : u[lfill-1];
			
			/*for( j=0; j<m; j++ )
				printf( "\t(%d %g) ", indx[j], u[j] );
			printf( "\n" ); */
		}
	} 
	
	//printf( "done sort\n" );
	
	// look after the diag element, this is ridiculously complex
	if( diag>=0 && diag<n )
	{
		ipos = diag BLOCK_V_SHIFT;
		val = 0.;
		for( j=0; j<BLOCK_SIZE*BLOCK_SIZE; ipos++ )
		{
			val += fabs(v[ipos]);
			if( !((++j)%BLOCK_SIZE) )
				ipos+=shift;
		}
		// if the diag was too small enough to be dropped then add it to the end of the list
		// this is a wee bit fiddly as we have to deal with some special cases/
		
		// case 1 : it has certainly been included
		if( val>drop_min )
			p = -1;
		// case 2 : it would have been dropped
		else if( val<=tol )
		{
			if( m<lfill )
			{
				p = m;
				m++;
			}
			else
				p = lfill-1;
		}
		// case 2 : we cannot be sure that it hasn't been dropped
		// this arrises when the value of the diag element is equal to that of
		// the smallest value in the list on nz entries to keep. There may
		// be more than one entry with this value in the array, so we must check
		// in case it was dropped
		else
		{
			p = m-1;
			k = 0;
			// search the end of
			while( !k && u[p]==u[m-1] )
			{
				if( indx[p] == diag )
					k=1;
				else
					p--;
			}
			
			// it has been included
			if( k )
				p = -1;
			// it has been dropped
			else
			{
				p = m;
				if( p==lfill )
					p--;
				else
					m++;
			}
		}
		if( p>=0 )
			indx[p] = diag;
	}
	
	// sort the indices
	heapsort_int( m, indx );
	
	// pack the entries into the start of v
	fill = (m<lfill) ? m : lfill; 
	from = 0;
	to = 0;
	for( p=0; p<BLOCK_SIZE; p++ )
	{
		to   = p*(fill BLOCK_V_SHIFT);
 		j    = (m<lfill) ? 0 : m - lfill;
		for( i=0; i<fill; i++, j++)
		{
			pos = (indx[j] BLOCK_V_SHIFT) + from;
			for( k=0; k<BLOCK_SIZE; k++, to++, pos++ )
				v[to] = v[pos];
		}
		from += (n BLOCK_V_SHIFT);
	}
	j = (m<lfill) ? 0 : m - lfill;
	for( i=0; i<fill; i++, j++ )
		indx[i] = indx[j];
	
	// return
	return fill;
}  

