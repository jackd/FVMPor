#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "linalg.h"
#include "benlib.h"
#include "ben_mpi.h"
#include "linalg_mpi.h"
#include "linalg_dense_mpi.h"
#include "fileio.h"
#include "gmres.h"
#include "MR.h"
#include "precon.h"

int deflate( Tmtx_ptr Hbar, Tmtx_dist_ptr Q, int k, Tgmres_ptr run, int root );

/******************************************************************************************
 *
 *   gmres()
 *   
 *   Vanilla flavoured right hand preconditioned GMRES.
 *
 ******************************************************************************************/
void gmres( Tmtx_CRS_dist_ptr A, Tvec_dist_ptr b, Tvec_dist_ptr x, Tgmres_ptr run, Tprecon_ptr precon, int root )
{
	int j, m, converged=0, restarts=0, K=0, block;
	int thisroot=0;
	double *y=NULL;
	double beta, normb, error;
	Tmtx Hbar, Htemp;
	Tvec_dist r, w, dx;
	Tmtx_dist Q;
	
	/* 
	 *  initialise variables 
	 */
	
	// am I the root process?
	if( root==A->This.this_proc )
		thisroot = 1;
	
	if( run->diagnostic && thisroot )
		fprintf( run->diagnostic, "Root process P%d starting GMRES run with root %d\n\n", A->This.this_proc, root );
	else if( run->diagnostic )
		fprintf( run->diagnostic, "Slave process P%d starting GMRES run with root %d\n\n", A->This.this_proc, root );
	
	// record the start time
	run->time_gmres = MPI_Wtime();
	
	// assorted variables
	m = run->dim_krylov;
	block = A->mtx.block;
	run->restarts = 0;
	
	// matrices
	Hbar.init = Htemp.init = Q.init = Q.mtx.init = 0;
	mtx_dist_init( &Q, &A->This, A->mtx.nrows, m+1, block, A->vtxdist );
	if( thisroot )
		mtx_init( &Hbar, m+1, m );
	
	// vectors
	r.init = w.init = dx.init = 0;
	y = (double *)calloc( m+1, sizeof(double) );
	vec_dist_init_vec( &r, b );
	vec_dist_init_vec( &w, b );
	vec_dist_init_vec( &dx, b );
		
	// calculate initial residual
	beta = mtx_CRS_dist_residual( A, x, b, &r );
	normb = vec_dist_nrm2( b );
	if( !(error=beta) || (error=beta/normb)<run->tol )
	{
		run->restarts = 0;
		run->time_gmres = MPI_Wtime() - run->time_gmres;
		if( run->diagnostic )
			fprintf( run->diagnostic, "The intial guess at the solution gives convergence with error = %g\n\n", error );		
		return;
	}
	y[0] = beta;
	
	// setup q1 : q1 = r/||r||
	vec_dist_scale( &r, 1./beta );
	mtx_dist_insert_col( &Q, 0, &r );	

	if( run->diagnostic )
		fprintf( run->diagnostic, "Intial residual has norm %g\n\n", beta );
		
	/* loop through restarts */
	while( restarts<=run->max_restarts && !converged )
	{
		if( run->diagnostic )
			fprintf( run->diagnostic, "RESTART %d\t", restarts );
		
		/* 
		 *  Arnoldi 
		 */
		
		// form the orthogonal Krylov basis using the arnoldi method
		j = arnoldi( A, m, 0, &Q, &Hbar, &w, precon, run->reorthogonalise, root );
		
		run->j[restarts]=j;
		
		/* 
		 *	solve least squares Hbar*y=Qm'*r 
		 */
		 
		// find RHS vector and store in y
		mtx_dist_gemv( &Q, &r, NULL, beta, 0., 0, Q.mtx.nrows-1, 0, j, 'T', y );

		
		// solve the least squares problem
		if( thisroot )
		{
			// make backup of Hbar, as the solver alters Hbar
			mtx_copy( &Hbar, &Htemp);
			
			// use Givens rotations on Hbar to reduce to upper triangular form
			mtx_givens( &Htemp, K, y );
			
			// solve triangular system for y 
			dtrsm( "L", "U", "N", "N", j, 1, 1., Htemp.dat, m+1, y, m );
		}
		
		/* 
		 * find new x
		 */
		
		// send y to all Pid
		MPI_Bcast( y, m+1, MPI_DOUBLE, root, A->This.comm );
		
		// w = Qm*y
		mtx_dist_gemv( &Q, NULL, &w, 1., 0., 0, Q.mtx.nrows-1, 0, j-1, 'n', y );
		
		// dx = Mw
		if( precon )
			precon_apply( precon, &w, &dx );
		else
			vec_dist_copy( &w, &dx );
		
		// x = x + dx 
		vec_dist_axpy( &dx, x, 1. );
		
		/* 
		 *  find new residual 
		 */
		beta = mtx_CRS_dist_residual( A, x, b, &r );
		error = beta/normb;
		run->residuals[restarts]=beta;
		run->errors[restarts]=error;
		
		// test residual for convergence
		if( error<run->tol )
			converged=1;
		else
			restarts++;
		
		// output convergence information to file
		if( run->diagnostic )
		{
			if( converged )
				fprintf( run->diagnostic, "j = %d\tK=%d\terror = %g  \tresidual = %g\tCONVERGED\n", j, K, error, beta );
			else
				fprintf( run->diagnostic, "j = %d\tK=%d\terror = %g  \tresidual = %g\n", j, K, error, beta );
		}
		
		/*
		 *  setup for the next iteration
		 */
		if( !converged && restarts<=run->max_restarts )
		{ 
			// This is automatically zero since we have no deflation
			run->K[restarts] = 0;
			
			// setup Q, Hbar and y
			mtx_clear( &Hbar );
			mtx_dist_clear( &Q );

			vec_dist_scale( &r, 1./beta );
			mtx_dist_insert_col( &Q, 0, &r );
		}
	} 
  
	// save the number of restarts
	run->restarts=restarts;
	if( !converged )
		run->restarts--;  

	// clean up memory used by GMRES
	free( y );
	vec_dist_free( &r );
	vec_dist_free( &w );
	vec_dist_free( &dx );
	mtx_free( &Hbar );
	mtx_free( &Htemp );
	mtx_dist_free( &Q );
	
	// get the total time for the GMRES
	run->time_gmres = MPI_Wtime() - run->time_gmres;
}

/******************************************************************************************
 *
 *   gmresFH()
 *
 *   Right hand preconditioned GMRES with harmonic Ritz pairs caluclated at each restart.
 *   The eigenvectors from the harmonic Ritz pair are augmented to the front of the subspace
 *   used in the next restart.
 *
 ******************************************************************************************/
void gmresFH( Tmtx_CRS_dist_ptr A, Tvec_dist_ptr b, Tvec_dist_ptr x, Tgmres_ptr run, Tprecon_ptr precon, int root )
{
	int j, m, converged=0, restarts=0, K=0, k=run->k, block;
	int thisroot=0;
	double *y=NULL;
	double beta, normb, error;
	Tmtx Hbar, Htemp;
	Tvec_dist r, w, dx;
	Tmtx_dist Q;
	
	/* 
	 *  initialise variables 
	 */
	
	// am I the root process
	if( root==A->This.this_proc )
		thisroot = 1;
	
	// wait for everyone to arrive	
	MPI_Barrier( A->This.comm );
	
	// record the start time
	run->time_gmres = MPI_Wtime();
	
	// assorted variables
	m = run->dim_krylov;
	block = A->mtx.block;
	run->restarts = 0;
	
	// matrices
	Hbar.init = Htemp.init = Q.init = Q.mtx.init = 0;
	mtx_dist_init( &Q, &A->This, A->mtx.nrows, m+1, block, A->vtxdist );
	if( thisroot )
		mtx_init( &Hbar, m+1, m );
	
	// vectors
	r.init = w.init = dx.init = 0;
	y = (double *)calloc( m+1, sizeof(double) );
	vec_dist_init_vec( &r, b );
	vec_dist_init_vec( &w, b );
	vec_dist_init_vec( &dx, b );
	
	// calculate initial residual
	beta = mtx_CRS_dist_residual( A, x, b, &r );
	normb = vec_dist_nrm2( b );
	if( !(error=beta) || (error=beta/normb)<run->tol )
	{
		run->restarts = 0;
		run->time_gmres = MPI_Wtime() - run->time_gmres;
		if( run->diagnostic )
			fprintf( run->diagnostic, "The intial guess at the solution gives convergence with error = %g\n\n", error );		
		return;
	}
	y[0] = beta;
	
	// setup q1 : q1 = r/||r||
	vec_dist_scale( &r, 1./beta );
	mtx_dist_insert_col( &Q, 0, &r );	
	
	if( run->diagnostic )
		fprintf( run->diagnostic, "\n" );
	
	/* loop through restarts */
	while( restarts<=run->max_restarts && !converged )
	{		
		if( run->diagnostic )
			fprintf( run->diagnostic, "RESTART %d\t", restarts );
		
		/* 
		 *  Arnoldi 
		 */
		
		// form the orthogonal Krylov basis using the arnoldi method
		j = arnoldi( A, m, K, &Q, &Hbar, &w, precon, run->reorthogonalise, root );
		run->j[restarts]=j;
		
		//MPI_Finalize();
		//exit(1);
		
		/* 
		 *	solve least squares Hbar*y=Qm'*r 
		 */
		
		// find RHS vector and store in y
		mtx_dist_gemv( &Q, &r, NULL, beta, 0., 0, Q.mtx.nrows-1, 0, j, 'T', y );
		
		// solve the least squares problem
		if( thisroot )
		{
			// make backup of Hbar, as the solver alters Hbar
			mtx_copy( &Hbar, &Htemp);
			
			// use Givens rotations on Hbar to reduce to upper triangular form
			mtx_givens( &Htemp, K, y );
			
			// solve triangular system for y
			dtrsm( "L", "U", "N", "N", j, 1, 1., Htemp.dat, m+1, y, m );
		}
		
		/* 
		 * find new x
		 */
		
		// send y to all Pid
		MPI_Bcast( y, m+1, MPI_DOUBLE, root, A->This.comm );
		
		// w = Qm*y
		mtx_dist_gemv( &Q, NULL, &w, 1., 0., 0, Q.mtx.nrows-1, 0, j-1, 'n', y );
		
		// dx = Mw
		if( precon )
			precon_apply( precon, &w, &dx );
		else
			vec_dist_copy( &w, &dx );
		
		// x = x + dx 
		vec_dist_axpy( &dx, x, 1. );
		
		/* 
		 *  find new residual 
		 */
		beta = mtx_CRS_dist_residual( A, x, b, &r );
		error = beta/normb;

// TESTING 2006
// ------------
if (A->This.this_proc == 0) printf("abs = %g, rel = %g\n", beta ,error);
// ------------

		run->residuals[restarts]=beta;
		run->errors[restarts]=error;
		
		// test residual for convergence
		if( error<run->tol )
			converged=1;
		else
			restarts++;
		
		// output convergence information to file
		if( run->diagnostic )
		{
			if( converged )
				fprintf( run->diagnostic, "j = %d\tK=%d\terror = %g  \tresidual = %g\tCONVERGED\n", j, K, error, beta );
			else
				fprintf( run->diagnostic, "j = %d\tK=%d\terror = %g  \tresidual = %g\n", j, K, error, beta );
		}
		
		/*
		 *  setup for the next iteration
		 */
		if( !converged && restarts<=run->max_restarts )
		{
			int i;
			
			/* 
			 * deflate eigenvalues from Hm 
			 */
			
			if( k )
				K = deflate( &Hbar, &Q, k, run, root );
						
			// setup Q, Hbar and y
			if( thisroot )
			{
				for( i=K*(m+1); i<m*(m+1); i++ )
					Hbar.dat[i] = 0.;
			}
			for( i=(K+1)*Q.mtx.nrows; i<Q.mtx.nrows*Q.mtx.ncols; i++ )
				Q.mtx.dat[i] = 0.;	
			run->K[restarts]=K;
			
			vec_dist_scale( &r, 1./beta );
			if( !k )
				mtx_dist_insert_col( &Q, 0, &r );
		}
	} 
	
	// save the number of restarts
	run->restarts=restarts;
	if( !converged )
		run->restarts--;  
	
	// clean up memory used by GMRES
	free( y );
	vec_dist_free( &r );
	vec_dist_free( &w );
	vec_dist_free( &dx );
	mtx_free( &Hbar );
	mtx_free( &Htemp );
	mtx_dist_free( &Q );
	
	// get the total time for the GMRES
	run->time_gmres = MPI_Wtime() - run->time_gmres;
}

/******************************************************************************************
*
*
*
******************************************************************************************/
int arnoldi( Tmtx_CRS_dist_ptr A, int m, int K, Tmtx_dist_ptr Q, Tmtx_ptr Hbar, Tvec_dist_ptr w, Tprecon_ptr precon, int reorthogonalise, int root )
{
	int j, i, pos, thisroot=0;
	double hval;
	Tvec_dist q;
	Tvec_dist wtmp;
		
	// figure out if we are the root process
	if( root == A->This.this_proc )
		thisroot = 1;
	
	// initialise the q temp working vector
	q.init = 0;
	vec_dist_init_vec( &q, w );
	free( q.dat );
	wtmp.init = 0;
	vec_dist_init_vec( &wtmp, w );
	
	// build the orthogonal basis
	j=K;
	while( j<m )
	{	
		/*
		 *		find w = A*qj or A*M*qj if preconditioner is turned on
		 */
		
		// point to qj
		q.dat = Q->mtx.dat + Q->mtx.nrows*j;
		
		// apply preconditioner
		if( !precon )
		{
			mtx_CRS_dist_gemv( A, &q, w, 1., 0., 'n' );
		}
		else
		{
			precon_apply( precon, &q, &wtmp );
			mtx_CRS_dist_gemv( A, &wtmp, w, 1., 0., 'n' );
		}

		
		
		// orthogonalise against Kj
		pos=j*(m+1);
		for( i=0; i<=j; i++, pos++ )
		{
			q.dat = Q->mtx.dat + Q->mtx.nrows * i;
			hval = vec_dist_dot( w, &q );
			if( thisroot )
				Hbar->dat[pos] = hval;
			vec_dist_axpy( &q, w, -hval );
		}
		
		// reorthogonalise
		if( reorthogonalise )
		{
			pos=j*(m+1);
			for( i=0; i<=j; i++, pos++ )
			{
				q.dat = Q->mtx.dat + Q->mtx.nrows * i;
				hval = vec_dist_dot( w, &q );
				if( thisroot )
					Hbar->dat[pos] += hval;
				vec_dist_axpy( &q, w, -hval );
			}
		}
		
		// setup for lucky breakdown
		hval = vec_dist_nrm2( w );
		if( thisroot)
		{
			Hbar->dat[pos]=hval;
		}
		if( hval<1e-16 )
		{
			if( thisroot )
				fprintf( stdout, "\t------------ LUCKY BREAKDOWN! hval = %g on j = %d ----------------\n", hval, j );
			break;
		}
		
		// store qj+1
		q.dat = Q->mtx.dat + Q->mtx.nrows*(j+1);
		vec_dist_axpy( w, &q, 1./hval );		
		
		// increment column counter
		j++;
	}
	
	// free up temp vector
	q.dat = NULL;
	vec_dist_free( &q );
	vec_dist_free( &wtmp );
		
	if( thisroot )
	{
		;//mtx_print( stdout, Hbar );
	}
	
	return j;
}

/******************************************************************************************
 *
 *  deflate()
 *
 *  Calculate k harmonic Ritz pairs of the smallest eigenvalues of A given Hbar
 *
 *
 *  POST : returns the dimension of the output matrix, may be greater than k depending on
 *	whether or not we have complex eigenvalues. returns 0 if failure
 *	X is a matrix with eignevectors as columns, if complex eigenvectors then consectutive
 *	columns hold the real and imaginary parts.
 *	lambda is tridiagonal matrix containing eigenvalues A*X=X*Lambda
 *
 ******************************************************************************************/
int deflate( Tmtx_ptr Hbar, Tmtx_dist_ptr Q, int k, Tgmres_ptr run, int root )
{
	int m, mm, i, j, pos, posm, p, thisroot=0, info;
	int *ipiv;
	double *f, *mags, *V, *lambdar, *lambdai, *dtmp, *em;
	double alpha, hmm, betak;
	Tvec_dist pm, q;
	Tmtx H, X, Ok;
	Tmtx_dist Qk;
	
	/***************************************  
	 *		Initialise variables
	 ***************************************/

	// am I the root process?
	if( root==Q->This.this_proc )
		thisroot = 1;
	
	// make sure all structures are ready to go
	H.init = X.init = Ok.init = Qk.init = q.init = pm.init = 0;
	
	if( thisroot )
	{
		// dimensions of problem
		m=Hbar->nrows-1;
		mm=run->dim_krylov;
		
		// miscelaneous values
		hmm = Hbar->dat[m*(m+1)-1];
		
		// allocate memory 
		ipiv		= (int*)malloc( m*sizeof(double) );
		mags		= (double*)malloc( m*sizeof(double) );
		lambdar		= (double*)malloc( m*sizeof(double) );
		lambdai		= (double*)malloc( m*sizeof(double) );
		V		= (double*)calloc( m*m*2, sizeof(double) );
		dtmp		= (double*)malloc( m*m*sizeof(double) );
		em		= (double*)calloc( m, sizeof(double) );
		
		mtx_init( &H, m, m );
		mtx_init( &X, m, 2*k );
	}
	else
	{
		lambdai = lambdar = V = mags = NULL;
	}
	
	/*********************************************************************
	 *		Communicate variabls local to root to other Pid
	 *********************************************************************/
	MPI_Bcast( &m, 1, MPI_INT, root, Q->This.comm );
	MPI_Bcast( &mm, 1, MPI_INT, root, Q->This.comm );
	MPI_Bcast( &hmm, 1, MPI_DOUBLE, root, Q->This.comm );
	
	f = (double*)calloc( m, sizeof(double) );
	
	/*****************************************************************************
	 *  Calculate the smallest eigenvalues and corresponding eigenvectors for
	 *  the Harmonic Ritz problem. The eigenvalues are then orthonormalised
	 *  against one another. All of This is a serial operation of small dimension,
	 *  the parallel part of the algorithm comes later on when the eigenvectors for
	 *  the large system are calculated from the Ritz eigenvectors.
	 *****************************************************************************/	
	
	if( thisroot )
	{
		/***************************************
		*		find f
		***************************************/
		
		// initiailise f to em
		f[m-1]=1.;
		
		// find Hm^T
		for( j=0, pos=0; j<m; j++, pos+=m )
			dcopy( m, Hbar->dat + j, mm+1, H.dat + pos, 1 );
		
		// solve for f
		i = 1;
		dgesv_( &m, &i, H.dat, &m, ipiv, f, &m, &info );
		
		/***************************************
		 *		find H
		 ***************************************/
		
		// set up H=Hm
		for( j=0, pos=0, posm=0; j<m; j++, pos+=m, posm+=mm )
			dcopy( m, Hbar->dat + (posm+j), 1, H.dat + pos, 1 );
		
		em[m-1] = 1.;
		
		// perform rank-1 update to find H : H = Hm + beta^2*f*e_m^T
		dger( m, m, hmm*hmm, f, 1, em, 1, H.dat, m );
		
		/*************************************** 
		 *		Find harmonic Ritz pairs 
		 ***************************************/
		
		// make a backup copy of H as it gets altered by the LAPACK eigenvalue routine
		dcopy( m*m, H.dat, 1, dtmp, 1 );
		
		// find the eigenvalues and set k=0 if unable to calculate them
		if( !mtx_eigs( &H, lambdar, lambdai, V ) )
			return 0;
		
		// reset H
		dcopy( m*m, dtmp, 1, H.dat, 1 );
		
		/***************************************
		 *		Sort the k smallest eigenvalues 
		 ***************************************/
		
		// find magnitudes of eigenvalues, don't bother with sqrt since we are just interested in sorting them 
		for( i=0; i<m; i++ )
			mags[i]=lambdar[i]*lambdar[i]+lambdai[i]*lambdai[i];
				
		// sort the eigenvalues by magnitude, keeping ipiv as a permutation array
		for( i=0; i<m; i++ )
			ipiv[i]=i;
		heapsort_double_index( m, ipiv, mags);


for (i = 0; i < k; ++i)
    printf("\t%g ", sqrt(mags[i]));
printf("\n");

		
		/****************************************************************************************************** 
		 *  store the eigenvectors corresponding to the k smallest eigenvalues, and store them in
		 *  X, making sure that real and imaginary parts are stored where imaginary components exist
		 ******************************************************************************************************/
		for( j=0, i=0, p=0; i<k; i++ )
		{
			pos = ipiv[i+p];
			
			// is the eigenvalue complex?
			if(  lambdai[pos]!=0. )
			{
				// if so then we need to store the real and imaginary parts 
				// all complex eigenvalues come in complex conjugate pairs, so we 
				// must test to see which member of the pair we are refering to 
				if( lambdai[pos]>0. )
				{
					dcopy( m, &V[pos*m], 1, &X.dat[j*m], 1 );
					dcopy( m, &V[(pos+1)*m], 1, &X.dat[(j+1)*m], 1 );
				}
				else
				{
					dcopy( m, &V[(pos-1)*m], 1, &X.dat[j*m], 1 );
					dcopy( m, &V[pos*m], 1, &X.dat[(j+1)*m], 1 );
				}
				j+=2;
				p++;
			}
			// else it must be real
			else
			{				
				dcopy( m, &V[pos*m], 1, &X.dat[j*m], 1 );
				j++;
			}
		}
		// from now on j holds the value k+z, where z is the number of complex eigenvalues we are deflating
		
		/****************************************************************************************
		 *		perform GS orthogonalisation on eigenvectors stored as columns in X
		 *
		 *		a future project is to apply to Q using LAPACK stored as reflectors, instead
		 *		of converting reflectors to a full matrix before multiplication
		 ****************************************************************************************/
		
		// now we know j we can give definate dimensions to X
		X.nrows=m;
		X.ncols=j;
		X.dat = realloc( X.dat, j*m*sizeof(double) );
		
		mtx_QR( &X );
		mtx_QR_extractQ( &X );		
		
		/***************************************
		 *  find Omegak = X^T * H * X
		 ***************************************/
		
		// intitialise Omegak
		mtx_init( &Ok, j, j );	
		
		// V = transpose(X)*H 
		dgemm( "t", "n", j, m, m, 1., X.dat, m, H.dat, m, 0., V, j );
		
		// Omegak = V*X
		dgemm( "n", "n", j, j, m, 1., V, j, X.dat, m, 0., Ok.dat, j );
	}
	
	/******************************************************************************
	*   From now on things have to happen in parallel since we are working with
	*   eigenvectors from the large system (A).
	******************************************************************************/
	
	// let the slaves know how many eigenvectors we are dealing with
	MPI_Bcast( &j, 1, MPI_INT, root, Q->This.comm );
	
	// Broadcast the X data from root to the other Pid
	if( !thisroot )
		mtx_init( &X, m, j );
	MPI_Bcast( X.dat, m*j, MPI_DOUBLE, root, Q->This.comm );
	
	/***************************************
 	 *		find Qk = Qm *  X
 	 ***************************************/
	
	// This operation is carried out in parallel, since Qk and Qm are distributed
	Qk.init = 0;
	mtx_dist_init_explicit( &Qk, &Q->This, Q->mtx.nrows, j+1, Q->block, Q->vtxdist );
	
	// we can be cheeky since a (dist*local) product requires no communication 
	dgemm( "n", "n", Q->mtx.nrows, j, m, 1., Q->mtx.dat, Q->mtx.nrows, X.dat, m, 0., Qk.mtx.dat, Q->mtx.nrows );

	/******************************************************************
	 *  find p_{m} = beta*(q_{m+1} - beta*Q_{m}*f)
	 ******************************************************************/
	
	// p_m = q_{m+1}
	mtx_dist_get_col( Q, m, &pm );
	
	// let all Pid know what f stored locally on root is
	MPI_Bcast( f, m, MPI_DOUBLE, root, Q->This.comm );
	
	// p_m = beta*p_m - beta*beta*Q_m*f
	dgemv( "n", Q->mtx.nrows, m, -hmm*hmm, Q->mtx.dat, Q->mtx.nrows, f, 1, hmm, pm.dat, 1 );
	
	/******************************************************* 
	 *		find v(k+1) by orthogonalising pm against Vk 
	 *******************************************************/
	
	// setup working vector to point at columns of Qk
	q.init = 0;
	vec_dist_init_vec( &q, &pm );
	free( q.dat );
	
	// use MGS for orthogonalisation
	for( i=0; i<j; i++ )
	{
		q.dat = Qk.mtx.dat + Qk.mtx.nrows * i;
		alpha = -vec_dist_dot( &pm, &q );
		vec_dist_axpy( &q, &pm, alpha );
	}
		
	// normalise p_m
	betak = vec_dist_nrm2( &pm );
	vec_dist_scale( &pm, 1./betak );
	
	// copy p_m as column j+1 of Qk
	mtx_dist_insert_col( &Qk, j, &pm );
	
	// free up q, take care because it was used as a pointer into a matrix
	q.dat = NULL;
	vec_dist_free( &q );
		
	/************************************************************************************************ 
	 *	Form first (j+1)*j chunk of Hbar 
	 *
	 *  Hbar is stored on the root, and has all of the info it needs available on the
	 *  root.
	 ************************************************************************************************/
	
	if( thisroot )
	{
		// zero out the entries in Hbar 
		for( i=0; i<mm*(mm+1); i++ ) 
			Hbar->dat[i]=0.;
		
		// find X^{T}*f, use V as a temp variable
		dgemv( "t", m, j, 1., X.dat, m, f, 1, 0., V, 1 );
		
		// perform rank-1 update on \Omega_{k} = \Omega_{k} - beta^2*X'*f*s'
		dger( j, j, -hmm*hmm, V, 1, X.dat + (m-1), m, Ok.dat, j );
		
		// store This in Hbar 
		for( i=0, pos=0, p=0; i<j; i++, pos+=(mm+1), p+=j )
			dcopy( j, Ok.dat + p, 1, Hbar->dat + pos, 1 );
		
		// calculate  and store in Hbar betak*transpose(s)
		dcopy( j, X.dat + (m-1), m, Hbar->dat + j, mm+1 );
		dscal( j, betak, &Hbar->dat[j], m+1 );
	}
	
	
	/*************************************************** 
	 *	Copy over first j+1 columns of Q for restart 
	 ***************************************************/
	dcopy( Qk.mtx.nrows*(j+1), Qk.mtx.dat, 1, Q->mtx.dat, 1 );
	
	/***************************************************
	 *		Close up 
	 ***************************************************/
	vec_dist_free( &q );
	vec_dist_free( &pm );
	mtx_dist_free( &Qk );
	mtx_free( &X );

	free( f );

	if( thisroot )
	{
		//mtx_print( stdout, &H );
		//mtx_print( stdout, Hbar );
		mtx_free( &Ok );
		mtx_free( &H );
		free( mags );
		free( V );
		free( lambdai );
		free( lambdar );

// BUGFIX 2006
// -----------
                free( ipiv );
                free( dtmp );
                free( em );
// -----------

	}
	
	return j;
}

int GMRES_params_load( Tgmres_ptr parms, char *fname )
{
	FILE *fid;
	int Pid;
	char diagnostic[256];
	
	MPI_Comm_rank( MPI_COMM_WORLD, &Pid );
	
	//if(  Pid ) return 0;
	
	// open parameter file for input
	if( !(fid=fopen(fname,"r")) )
	{
		printf( "GMRES_params_load() : unable to read parameters from file %s\n\n", fname );
		return 0;
	}
	
	/***********************************************
  	 *	read the parameters from the input file
	 ***********************************************/
	
	if( !fgetvar( fid, "%d", &parms->dim_krylov ) )		return 0;
	if( !fgetvar( fid, "%d", &parms->k ) )				return 0;
	if( !fgetvar( fid, "%d", &parms->max_restarts ) )	return 0;
	if( !fgetvar( fid, "%lf", &parms->tol ) )			return 0;
	if( !fgetvar( fid, "%d", &parms->nrmcols ) )		return 0;
	if( !fgetvar( fid, "%d", &parms->reorthogonalise ) )return 0;
	if( !fgetvar( fid, "%s", diagnostic ) )				return 0;
	
	// setup the diagnostic stream
	if( Pid )
		parms->diagnostic = NULL;
	else if( !strcmp( diagnostic, "stdout" ) )
		parms->diagnostic = stdout;
	else if( !strcmp( diagnostic, "NULL" ) )
		parms->diagnostic = NULL;
	else
		if( (parms->diagnostic = fopen( diagnostic, "w" ))==NULL )
		{
			printf( "GMRES_params_load() : unable to open diagnostic stream %s specified in %s\n", diagnostic, fname );
			fclose( fid );
			return 0;
		}
	
	// allocate the memory needed for GMRES stats
	parms->residuals = (double *)malloc( (parms->max_restarts+1)*sizeof(double) );
	parms->errors = (double *)malloc( (parms->max_restarts+1)*sizeof(double) );
	parms->j = (int *)malloc( (parms->max_restarts+1)*sizeof(int) );
	parms->K = (int *)malloc( (parms->max_restarts+1)*sizeof(int) );
	
	// initialise other variables
	parms->restarts = 0;
	parms->time_precon = 0;
	parms->time_gmres = 0;
	parms->copied = 0;
	
	fclose( fid );
	return 1;
}

void GMRES_params_free( Tgmres_ptr parms )
{

// BUGFIX 2006
// -----------

// Tests (in this case) are harmless, but still unnecessary.  Search for BUGFIX
// to find examples of similar, but harmful tests in this code.

//	if( parms->residuals )
		free( parms->residuals );
//	if( parms->errors )
		free( parms->errors );
//	if( parms->j )
		free( parms->j );
//	if( parms->K )
		free( parms->K );
// -----------



	
	if( !parms->copied && parms->diagnostic && (parms->diagnostic!=stdout) )
		fclose( parms->diagnostic );
}
	
void GMRES_params_copy( Tgmres_ptr from, Tgmres_ptr to )
{
	to->copied = 1;
	to->dim_krylov = from->dim_krylov;
	to->max_restarts = from->max_restarts;
	to->tol = from->tol;
	to->restarts = from->restarts;
	to->nrmcols = from->nrmcols;
	to->k = from->k;
	to->reorthogonalise = from->reorthogonalise;
	to->time_precon = from->time_precon;
	to->time_gmres = from->time_gmres;
	to->diagnostic = from->diagnostic;
	to->residuals = (double *)malloc( (from->max_restarts+1)*sizeof(double) );
	to->errors = (double *)malloc( (from->max_restarts+1)*sizeof(double) );
	to->j = (int *)malloc( (from->max_restarts+1)*sizeof(int) );
	to->K = (int *)malloc( (from->max_restarts+1)*sizeof(int) );
}
