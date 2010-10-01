#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "indices.h"
#include "linalg_sparse.h"
#include "benlib.h"

char errmsg[256];

/******************************************************************************************************************* 
*       mtx_CRS_init()
*
*       initialise a CRS matrix
*******************************************************************************************************************/
void mtx_CRS_init( Tmtx_CRS_ptr A, int nrows, int ncols, int nnz, int block )
{
    if( A->init )
        mtx_CRS_free( A );
    
    A->init = 1;
    A->nrows = nrows;
    A->ncols = ncols;
    A->nnz = nnz;
    A->block = block;
    
    if( !nnz )
    {
        A->rindx = calloc( nrows+1, sizeof(int) );
        A->cindx = NULL;
        A->nz    = NULL;
        return;
    }
    
    A->cindx = malloc( sizeof(int)*nnz );
    A->rindx = malloc( sizeof(int)*(nrows+1));
    
    if( !block )
    {
        A->nz = calloc( nnz, sizeof(double) );
    }
    else
    {
        A->nz = calloc( nnz BLOCK_M_SHIFT, sizeof(double) );
    }
}

/******************************************************************************************************************* 
*       mtx_CRS_free()
*
*       free a CRS matrix : setting to unititialised
*******************************************************************************************************************/
void mtx_CRS_free( Tmtx_CRS_ptr A )
{
    if( !A->init )
        return;
    A->init = 0;
    A->nrows = 0;
    A->ncols= 0;
    A->nnz = 0;
    A->block = 0;

// BUGFIX 2006
// -----------

    // free(NULL) is perfectly legal: testing for it is unnecessary and (as
    // witnessed below) can lead to errors if you get the test backwards.

//  if( !A->rindx )
        free( A->rindx );
//  if( !A->cindx )
        free( A->cindx );
//  if( !A->nz )
        free( A->nz );

// -----------
    
    A->rindx = A->cindx = NULL;
    A->nz = NULL;
}

/******************************************************************************************************************* 
 *      mtx_CRS_copy()
 *
 *      copy a CRS matrix : to = from
 *******************************************************************************************************************/
void mtx_CRS_copy( Tmtx_CRS_ptr from, Tmtx_CRS_ptr to )
{
    ASSERT_MSG( from->init, "mtx_CRS_copy() : source matrix must be initialised" );

    mtx_CRS_init( to, from->nrows, from->ncols, from->nnz, from->block );
    
    memcpy( to->cindx, from->cindx, sizeof(int)*from->nnz );
    memcpy( to->rindx, from->rindx, sizeof(int)*(from->nrows+1) );
    if( !from->block )
        dcopy( from->nnz, from->nz, 1, to->nz, 1 );
    else
        dcopy( (from->nnz BLOCK_M_SHIFT), from->nz, 1, to->nz, 1 );
} 


//  these routines are exactly the same as the ones above, but the work for CCS matrices


/******************************************************************************************************************* 
*       mtx_CCS_init()
*
*       initialise a CCS matrix
*******************************************************************************************************************/
void mtx_CCS_init( Tmtx_CCS_ptr A, int nrows, int ncols, int nnz, int block )
{
    if( A->init )
        mtx_CCS_free( A );
    A->init = 1;
    A->nrows = nrows;
    A->ncols = ncols;
    A->nnz = nnz;
    A->block = block;
    
    if( !nnz )
    {
        A->cindx = calloc( ncols+1, sizeof(int) );
        A->rindx = NULL; 
        A->nz    = NULL;
        return;
    }
    
    A->cindx = malloc( sizeof(int)*(ncols+1) );
    A->rindx = malloc( sizeof(int)*nnz );
    
    if( !block )
        A->nz = calloc( nnz, sizeof(double) );
    else
        A->nz = calloc( nnz BLOCK_M_SHIFT, sizeof(double) );
}

/*******************************************************************************************************************
    free a CCS matrix
*******************************************************************************************************************/
void mtx_CCS_free( Tmtx_CCS_ptr A )
{
    if( !A->init )
        return;
    A->init = 0;
    A->nrows = 0;
    A->ncols= 0;
    A->nnz = 0;
    A->block = 0;

// BUGFIX 2006
// -----------

    // free(NULL) is perfectly legal: testing for it is unnecessary and (as
    // witnessed below) can lead to errors if you get the test backwards.
    
//  if( !A->rindx )
        free( A->rindx );
//  if( !A->cindx )
        free( A->cindx );
//  if( !A->nz )
        free( A->nz );

// -----------

    A->rindx = A->cindx = NULL;
    A->nz = NULL;
}

/*******************************************************************************************************************
    copy a CCS matrix
*******************************************************************************************************************/
void mtx_CCS_copy( Tmtx_CCS_ptr from, Tmtx_CCS_ptr to )
{
    mtx_CCS_init( to, from->nrows, from->ncols, from->nnz, from->block );
    
    memcpy( to->cindx, from->cindx, sizeof(int)*(from->ncols+1) );
    memcpy( to->rindx, from->rindx, sizeof(int)*from->nnz );
    if( !from->block )
        dcopy( from->nnz, from->nz, 1, to->nz, 1 );
    else
        dcopy( (from->nnz BLOCK_M_SHIFT), from->nz, 1, to->nz, 1 );
}

/*******************************************************************************************************************
    make a CCS identity matrix
*******************************************************************************************************************/
void mtx_CCS_eye( Tmtx_CCS_ptr A, int dim, int block )
{
    if( !A->init || A->block!=block || A->nrows!=A->nrows || A->nrows!=dim || A->nnz<dim )
        mtx_CCS_init( A, dim, dim, dim, block );
    
    fprintf( stdout, "you dick, make mtx_CCS_eye() work for all blocks!\n" );
    
    if( A->block )
    {
        int i; 
        for( i=0; i<4*dim; i+=4 )
        {
            A->nz[i]    = 1.;
            A->nz[i+3]  = 1.;
        }
        for( i=0; i<dim; i++ )
        {
            A->rindx[i] = i;
            A->cindx[i] = i;
        }
        A->cindx[i] = i;
    }
    else
    {
        int i;
        for( i=0; i<dim; i++ )
        {
            A->nz[i] = 1.;
            A->rindx[i] = i;
            A->cindx[i] = i;
        }
        A->cindx[i] = i;
    }
    A->nnz = dim;
}

/********************************************************************************************
*       mtx_yale_index_swap()
*
*       swaps the cindx and rindx pointers along with the nrows and ncols values
*           - if A is CCS then A will become a A^(transpose) in CRS
*           - if A is CRS then A will become a A^(transpose) in CCS
********************************************************************************************/
void mtx_yale_index_swap( Tmtx_yale_ptr A  )
{
    int *tmp_ptr;
    int tmp;
    
#ifdef DEBUG
    ASSERT_MSG( A->init, "mtx_yale_index_swap() : matrix must be initialised\n"  );
#endif
    
    // swap indices
    tmp_ptr = A->rindx;
    A->rindx = A->cindx;
    A->cindx = tmp_ptr;
    
    // swap dimensions
    tmp = A->nrows;
    A->nrows = A->ncols;
    A->ncols = tmp;
}

/*******************************************************************************************************************
    computes the scalar product of a matrix and adds that to another matrix
 
    Y = a*X + Y;
 
    where a=scalar and X and Y are matrices of the same dimension. It is the caller's responsibility
    to ensure that X and Y have been properly initialised. Y is overwritten and more memory shall be
    allocated if needed to allow for extra fill in as a result of the operation. If you wish to add
    two matrices with the same sparsity patterns use the BLAS daxpy routine directly on the ->nz arrays
    of the matrices in question.
*******************************************************************************************************************/
void mtx_CCS_axpy( double a, Tmtx_CCS_ptr X, Tmtx_CCS_ptr Y )
{
    int i, pos=0, col=0, nX, nY, n;
    int *index, *rindx, *cindx, *tempr;
    double *column, *nz;
    
    // simple check that the user has compatable and valid matrices
    ASSERT( X->nrows==Y->nrows && X->ncols==Y->ncols && X->init && Y->init );
    
    // if a is zero then there is no work to do!
    if( !a )
        return;
    
    // allocate memory for the work arrays
    column = (double*)malloc( sizeof(double)*X->nrows );
    index = (int*)malloc( sizeof(int)*X->nrows );
    cindx = (int*)malloc( sizeof(int)*X->nrows+1 );
    rindx = (int*)malloc( sizeof(int)*(X->nnz+Y->nnz) );
    tempr = (int*)malloc( sizeof(int)*(X->nrows*2) );
    nz = (double*)malloc( sizeof(double)*(X->nnz+Y->nnz) );
    
    // perform the addition, one column at a time
    cindx[0] = 0;
    for( col=0; col<X->ncols; col++ )
    {
        // calculate the relevant entries in the work array
        n = 0;
        nX = X->cindx[col+1]-X->cindx[col];
        nY = Y->cindx[col+1]-Y->cindx[col];
        for( i=Y->cindx[col]; i<Y->cindx[col+1]; i++  )
        {  
            // zero out the contributions due to Y
            column[Y->rindx[i]] = 0.;
        }
        for( i=X->cindx[col]; i<X->cindx[col+1]; i++ )
        {
            // find the contribution due to X
            column[X->rindx[i]] = a*X->nz[i];
            index[n++] = X->rindx[i];
        }
        for( i=Y->cindx[col]; i<Y->cindx[col+1]; i++  )
        {
            // now add the contribution due to Y
            column[Y->rindx[i]] += Y->nz[i];
            index[n++] = Y->rindx[i];
        }
        
        // sort the indices using a merge sort as the set of indices
        // is actually two sorted arrays appended to one-annother arrays
        /*
            IF THERE IS A BUG, CHECK THIS OUT FIRST, try a heapsort
         */
        merge( index, tempr, 0, nX, nX+nY-1);
        
        // remove duplicate entries and store the new row indices and 
        // nz entries
        nX = 0;
        i = 0;
        while( i<n )
        {
            nz[pos] = column[tempr[i]];
            rindx[pos] = index[i++];
            if( i<n && rindx[pos]==index[i] )
                i++;
            pos++;
        }
        
        // update the column indices
        cindx[col+1]=pos;
    }
    
    // resize arrays
    rindx = (int *)realloc( rindx, sizeof(int)*pos );
    nz = (double *)realloc( nz, sizeof(double)*pos );
    
    // now transfer them over to Y
    free( Y->cindx );
    free( Y->rindx );
    free( Y->nz );
    Y->cindx = cindx;
    Y->rindx = rindx;
    Y->nz = nz;
    Y->nnz = pos;
    
    // free memory for work arrays
    free( column );
    free( index );
    free( tempr );
}

/******************************************************************************************************************
*   mtx_CCS_getcol_sp()
*
*   store column col from A in the sparse vector x
******************************************************************************************************************/
void mtx_CCS_getcol_sp( Tmtx_CCS_ptr A, Tvec_sp_ptr x, int col )
{
    int nnz;
    
#ifdef DEBUG
    ASSERT_MSG( col>=0 && col<A->ncols, "mtx_CCS_getcol_sp() : column value out of range." );
    ASSERT_MSG( A->init, "mtx_CCS_getcol_sp() : A not initialised.");
#endif
    
    // find the number of nonzero elements in This column
    nnz = A->cindx[col+1] - A->cindx[col];
    
    // if there are no nz elements, make the vector empty and leave
    if( !nnz )
    {
        x->init = 1;
        x->nnz = 0;
        x->n = A->nrows;
        x->block = A->block;
        x->nz = NULL;
        return;
    }
    
    // initialise vector
    vec_sp_init( x, A->nrows, nnz, A->block );
    
    // copy over information
    memcpy( x->indx, A->rindx + A->cindx[col], sizeof(int)*nnz );
    memcpy( x->nz,   A->nz    + A->cindx[col], sizeof(double)*nnz );
}

/*******************************************************************************************************************
    computes the scalar product of a matrix and adds that to another matrix
 
    Y = a*X + Y;
 
    where a=scalar and X and Y are matrices of the same dimension. It is the caller's responsibility
    to ensure that X and Y have been properly initialised. Y is overwritten and more memory shall be
    allocated if needed to allow for extra fill in as a result of the operation. If you wish to add
    two matrices with the same sparsity patterns use the BLAS daxpy routine directly on the ->nz arrays
    of the matrices in question.
 *******************************************************************************************************************/
void mtx_CCSB_axpy( double a, Tmtx_CCS_ptr X, Tmtx_CCS_ptr Y )
{
    int i,  pos=0, col=0, nX, nY, n;
    int *index, *rindx, *cindx, *tempr;
    double *column, *nz, *rp1, *rp2;
    
    // simple check that the user has compatable and valid matrices
    ASSERT_MSG( X->nrows==Y->nrows && X->ncols==Y->ncols && X->init && Y->init, "mtx_CCSB_axpy() : matrix dimensions noooo good" );
    ASSERT_MSG( X->block && Y->block, "mtx_CCSB_axpy() : matrices must be block matrices" );
    
    // if a is zero then there is no work to do!
    if( !a )
        return;
    
    // allocate memory for the work arrays
    column = (double*)malloc( (sizeof(double)*X->nrows ) BLOCK_M_SHIFT);
    index = (int*)malloc( (sizeof(int)*X->nrows BLOCK_M_SHIFT) );
    cindx = (int*)malloc( (sizeof(int)*(X->nrows+1) BLOCK_M_SHIFT) );
    rindx = (int*)malloc( (sizeof(int)*(X->nnz+Y->nnz) BLOCK_M_SHIFT) );
    tempr = (int*)malloc( (sizeof(int)*(X->nrows*2) BLOCK_M_SHIFT) );
    nz = (double*)malloc( (sizeof(double)*(X->nnz+Y->nnz) BLOCK_M_SHIFT) );
    
    // perform the addition, one column at a time
    cindx[0] = 0;
    for( col=0; col<X->ncols; col++ )
    {
        // calculate the relevant entries in the work array
        n = 0;
        nX = X->cindx[col+1]-X->cindx[col];
        nY = Y->cindx[col+1]-Y->cindx[col];
        for( i=Y->cindx[col]; i<Y->cindx[col+1]; i++  )
        {  
            // zero out the contributions due to Y
            /* ---------------------------------------
                column[Y->rindx[i]] = 0.;
            --------------------------------------- */
            rp1 = column + (Y->rindx[i] BLOCK_M_SHIFT);
            BLOCK_M_SET( 0., rp1 );
        }
        for( i=X->cindx[col]; i<X->cindx[col+1]; i++ )
        {
            // find the contribution due to X
            /* ---------------------------------------
                column[X->rindx[i]] = a*X->nz[i];   
            --------------------------------------- */
            rp1 = X->nz + (X->rindx[i] BLOCK_M_SHIFT);
            rp2 = column + ( X->rindx[i]  BLOCK_M_SHIFT);
            BLOCK_M_COPY( rp1, rp2 );
            BLOCK_M_SMUL( a, rp2 );
            index[n++] = X->rindx[i];
        }
        for( i=Y->cindx[col]; i<Y->cindx[col+1]; i++  )
        {
            // now add the contribution due to Y
            /* ---------------------------------------
                column[Y->rindx[i]] += Y->nz[i];
            --------------------------------------- */
            rp1 = Y->nz + (i BLOCK_M_SHIFT);
            rp2 = column + (Y->rindx[i]  BLOCK_M_SHIFT);            
            BLOCK_M_ADD( rp2, rp2, rp1 );
            index[n++] = Y->rindx[i];
        }
        
        // sort the indices using a merge sort as the set of indices
        // is actually two sorted arrays appended to one-annother arrays
        merge( index, tempr, 0, nX, nX+nY-1);
        
        // remove duplicate entries and store the new row indices and 
        // nz entries
        nX = 0;
        i=0;
        while( i<n )
        {
            /* ---------------------------------------
                nz[pos] = column[tempr[i]]; 
            --------------------------------------- */
            rp1 = nz + ( pos BLOCK_M_SHIFT );
            rp2 = column + ( index[i] BLOCK_M_SHIFT );
            BLOCK_M_COPY( rp2, rp1 );
            rindx[pos] = index[i++];
            if( i<n && rindx[pos]==index[i] )
                i++;
            pos++;
        }

        // update the column indices
        cindx[col+1]=pos;
    }
    
    // resize arrays
    rindx = (int *)realloc( rindx, sizeof(int)*pos );
    nz = (double *)realloc( nz, (sizeof(double)*pos) BLOCK_M_SHIFT );
    
    // now transfer them over to Y
    free( Y->cindx );
    free( Y->rindx );
    free( Y->nz );
    Y->cindx = cindx;
    Y->rindx = rindx;
    Y->nz = nz;
    Y->nnz = pos;
    
    // free memory for work arrays
    free( column );
    free( index );
    free( tempr );
}

/*******************************************************************************************************************
 *  mtx_CRS_transpose()
 *  
 *  B = A^T
 *******************************************************************************************************************/
void mtx_CRS_transpose( Tmtx_CRS_ptr A, Tmtx_CRS_ptr B )
{
    int *p;
    
    /* 
     *      check that user has passed valid info
     */
    ASSERT_MSG( A->init, "mtx_CRS_transpose() : source matrix not initialised" );
    
    /*
     *      initialise variables
     */
    mtx_CCS_init( (Tmtx_CCS_ptr)B, A->nrows, A->ncols, A->nnz, A->block );
    p = (int *)malloc( sizeof(int)*A->nnz );
    
    /*
     *      convert to CCS
     */
    index_CRS_to_CCS( A->cindx, A->rindx, B->cindx, B->rindx, p, A->nnz, A->nrows, A->ncols );
    
    /*
     *      reorder the elements of the matrix
     */
    if( !A->block )
    {
        permute( A->nz, B->nz, p, A->nnz, 1 );
    }
    else
    {
        permuteB( A->nz, B->nz, p, A->nnz, 1 );
        block_transpose( B->nz, B->nnz );
    }
    
    /*
     *      convert back to CRS while transposed
     */
    mtx_yale_index_swap( B );
    
    /*
     *      clean up and go home
     */ 
    free( p );
}

/*******************************************************************************************************************
*   mtx_CCS_transpose()
*   
*   B = A^T
*******************************************************************************************************************/
void mtx_CCS_transpose( Tmtx_CCS_ptr A, Tmtx_CCS_ptr B )
{
    int *p;
    Tmtx_CCS C;
    
#ifdef DEBUG
    ASSERT_MSG( A->init , "mtx_CCS_transpose() : source matrix not initialised" );
#endif
    
    /*
     *      initialise variables
     */

    mtx_CCS_init( B, A->ncols, A->nrows, A->nnz, A->block );
    
    // set C to be an exact copy of A
    mtx_CCS_copy( A, &C );
    p = (int *)malloc( sizeof(int)*A->nnz );
    
    /*
     *      convert C to CRS transpose of A
     */
    mtx_yale_index_swap( &C );
    
    /*
     *      convert to CCS
     */
    index_CRS_to_CCS( C.cindx, C.rindx, B->cindx, B->rindx, p, C.nnz, C.nrows, C.ncols );
    
    /*
     *      reorder the elements of the matrix
     */
    if( !A->block )
    {
        permute( A->nz, B->nz, p, A->nnz, 1 );
    }
    else
    {
        permuteB( A->nz, B->nz, p, A->nnz, 1 );
        block_transpose( B->nz, B->nnz );
    }
    
    /*
     *      clean up and go home
     */
    mtx_CCS_free( &C );
    free( p );
}

/*******************************************************************************************************************
 *
 *
 *
 *******************************************************************************************************************/
void mtx_CCS_to_CRS( Tmtx_CCS_ptr A, Tmtx_CRS_ptr B )
{
    int *p;
    Tmtx_CCS C;

// BUGFIX 2006
// -----------
        C.init = 0;
// -----------
    
#ifdef DEBUG
    ASSERT_MSG( A->init , "mtx_CCS_transpose() : source matrix not initialised" );
#endif
    
    // initialise variables
    mtx_CCS_init( B, A->ncols, A->nrows, A->nnz, A->block );
    
    // set C to be an exact copy of A
    mtx_CCS_copy( A, &C );
    p = (int *)malloc( sizeof(int)*A->nnz );
    
    //  convert C to CRS transpose of A
    mtx_yale_index_swap( &C );
    
    //  convert to CCS
    index_CRS_to_CCS( C.cindx, C.rindx, B->cindx, B->rindx, p, C.nnz, C.nrows, C.ncols );
    
    // reorder the elements of the matrix
    if( !A->block )
        permute( A->nz, B->nz, p, A->nnz, 1 );
    else
        permuteB( A->nz, B->nz, p, A->nnz, 1 );
    
    // flip it around
    mtx_yale_index_swap( B );
    
    // clean up and go home
    mtx_CCS_free( &C );
    free( p );
}

/*******************************************************************************************************************
*
*
*
*******************************************************************************************************************/
void mtx_CRS_to_CCS( Tmtx_CRS_ptr A, Tmtx_CCS_ptr B )
{
    int *p;
    
#ifdef DEBUG
    ASSERT_MSG( A->init, "mtx_CRS_to_CCS_convert() : source matrix A not initialised" );
#endif
    
    // initialise variables
    p = (int *)malloc( sizeof(int)*A->nnz );
    mtx_CCS_init( B, A->nrows, A->ncols, A->nnz, A->block );
    
    // find reordering for transpose
    index_CRS_to_CCS( A->cindx, A->rindx, B->cindx, B->rindx, p, A->nnz, A->nrows, A->ncols );
    
    // reorder the elements of the matrix
    if( !A->block )
        permute( A->nz, B->nz, p, A->nnz, 1 );
    else
        permuteB( A->nz, B->nz, p, A->nnz, 1 );
    
    // clean up and go home
    free( p );
}

/**********************************************************
 
**********************************************************/
void vec_sp_init( Tvec_sp_ptr v, int n, int nnz, int block )
{
    v->init = 1;
    v->block = block;
    v->n = n;
    v->nnz = nnz;
    if( !block )
        v->nz = (double *)malloc( sizeof(double)*nnz );
    else
        v->nz = (double *)malloc( sizeof(double)*(nnz BLOCK_V_SHIFT) );
    v->indx = (int *)malloc( sizeof(int)*nnz ); 
    ASSERT_MSG( v->nz != NULL, "vec_sp_init() : malloc error" );
}

/**********************************************************
    vec_sp_axpy()
 
    add two sparse vectors : y = y + ax 
**********************************************************/
void vec_sp_axpy( double a, Tvec_sp_ptr x, Tvec_sp_ptr y )
{
    int i, n, pos, k, ind;
    double *tmp;
    int *indx, *tmpindx;
    
#ifdef DEBUG
    ASSERT_MSG( x->init && y->init, "vec_sp_axpy() : vectors not initialised" );
    ASSERT_MSG( x->n == y->n, "vec_sp_axpy() : vectors must be same length");
    ASSERT_MSG( x->block == y->block, "vec_sp_axpy() : vectors must be both block/scalar");
#endif
    
    /*------------------------------------------
        SCALAR ENTRIES
    ------------------------------------------*/
    if( !x->block )
    {
        /*------------------------------------------
        initialise variables
        ------------------------------------------*/
        n = x->n;
        tmp = (double *)malloc( sizeof(double)*n );
        indx = (int *)malloc( sizeof(int)*(x->nnz+y->nnz) );
        tmpindx = (int *)malloc( sizeof(int)*(x->nnz+y->nnz) );
        
        // only need to zero out parts of temp vector that correspond to nz entries in y
        for( i=0; i<y->nnz; i++ )
        {
            tmp[y->indx[i]] = 0.;
        }
        
        /*------------------------------------------
            perform addition
            ------------------------------------------*/
        // first of all, find the sum in the temp vector
        pos = 0;
        for( i=0; i<x->nnz; i++, pos++ )
        {
            k = x->indx[i];
            tmp[k] = a*x->nz[i];
            indx[pos] = k;
        }
        for( i=0; i<y->nnz; i++, pos++ )
        {
            k = y->indx[i];
            tmp[k] += y->nz[i];
            indx[pos] = k;
        }
        
        // sort the indices
        /*
         IF THERE IS A BUG, CHECK THIS OUT FIRST, try a heapsort
         */
        merge( indx, tmpindx, 0, x->nnz, x->nnz+y->nnz-1);
        
        // do some organisation
        vec_sp_free( y );
        vec_sp_init( y, x->n, pos, x->block  );
        
        // now collect everything into sparse format
        k = 0;
        i = 0;
        while( i<pos )
        {
            // add the element if it is nz
            ind=indx[i];
            if( tmp[ind] )
            {
                y->nz[k] = tmp[ind];
                y->indx[k++] = ind;
            }
            i++;
            
            // test to see if we have doubled up at all
            if( indx[i] == ind )
                i++;
        }
        
        // now tidy up y
        y->nnz = k;
        y->nz = (double *)realloc( y->nz, sizeof(double)*k );
        y->indx = (int *)realloc( y->indx, sizeof(int)*k );
    }
    /*------------------------------------------
        BLOCK ENTRIES
    ------------------------------------------*/
    else
    {
        double *dp, *nzp;
        
        /*------------------------------------------
        initialise variables
        ------------------------------------------*/
        n = x->n;
        tmp = (double *)calloc( sizeof(double), (n BLOCK_V_SHIFT) );
        indx = (int *)malloc( sizeof(int)*(x->nnz+y->nnz));
        tmpindx = (int *)malloc( sizeof(int)*(x->nnz+y->nnz) );
        
        /*------------------------------------------
            perform addition
        ------------------------------------------*/
        // first of all, find the sum in the temp vector
        pos = 0;
        nzp = x->nz;
        for( i=0; i<x->nnz; i++, pos++, nzp += (1 BLOCK_V_SHIFT) )
        {
            dp = tmp + (x->indx[i] BLOCK_V_SHIFT);
            BLOCK_V_ACOPY( a, nzp, dp );
            indx[pos] = x->indx[i];
        }
        nzp = y->nz;
        for( i=0; i<y->nnz; i++, pos++, nzp += (1 BLOCK_V_SHIFT) )
        {
            dp = tmp + (y->indx[i] BLOCK_V_SHIFT);
            BLOCK_V_ADD( dp, dp, nzp );
            indx[pos] = y->indx[i];
        }
        
        // sort the indices
        /*
         IF THERE IS A BUG, CHECK THIS OUT FIRST, try a heapsort
         */
        merge( indx, tmpindx, 0, x->nnz, x->nnz+y->nnz-1);
        
        // do some organisation
        vec_sp_free( y );
        vec_sp_init( y, x->n, pos, x->block  );
        
        // now collect everything into sparse format
        k = 0;
        i = 0;
        while( i<pos )
        {
            // add the element if it is nz
            ind=indx[i];
            dp = tmp + (ind BLOCK_V_SHIFT);
            if( BLOCK_V_ISNZ( dp ) )
            {
                nzp = y->nz + (k BLOCK_V_SHIFT);
                BLOCK_V_COPY( nzp, dp );
                y->indx[k++] = ind;
            }
            i++;
            // test to see if we have doubled up at all
            if( indx[i] == ind )
                i++;
        }
        
        // now tidy up y
        y->nnz = k;
        y->nz = (double *)realloc( y->nz, sizeof(double)*(k BLOCK_V_SHIFT) );
        y->indx = (int *)realloc( y->indx, sizeof(int)*k );
    }
    
    /*------------------------------------------
        clean up
    ------------------------------------------*/
    free(tmpindx);
    free(indx);
    free(tmp);
}

/*********************************************************
    gather a dense vector into a sparse vector
 
    x -> y
*********************************************************/
void vec_sp_gather( Tvec_sp_ptr y, double *x, int n, int block )
{
    int i, pos;
    
    fprintf( stderr, "Goose, vec_sp_gather() don't pass block, check y for compatability\n" );
    
    // intialise target vector
    if( y->init )
        vec_sp_free( y );
    vec_sp_init( y, n, n, block );
    
    // scalar entries
    if( !block )
    {   
        pos = 0;
        for( i=0; i<n; i++ )
        {
            if( x[i] )
            {
                y->nz[pos] = x[i];
                y->indx[pos] = i;
                pos++;
            }
        }
        y->nnz  = pos;
        y->nz   = (double*)realloc( y->nz, pos*sizeof(double) );
        y->indx = (int*)realloc( y->indx, pos*sizeof(int) );
    }
    // block entries
    else
    {
        double *xp = x, *nzp = y->nz;
        
        pos = 0;
        for( i=0; i<n; i++, xp+=(1 BLOCK_V_SHIFT) )
        {
            if( BLOCK_V_ISNZ( xp ) )
            {
                BLOCK_V_COPY( xp, nzp );
                y->indx[pos] = i;
                pos++;
                nzp += (1 BLOCK_V_SHIFT);
            }
        }
        y->nnz  = pos;
        y->nz   = (double*)realloc( y->nz, (pos BLOCK_V_SHIFT)*sizeof(double) );
        y->indx = (int*)realloc( y->indx, pos*sizeof(int) );
    }
}

/********************************************************
    print out a sparse vector
*********************************************************/
void vec_sp_print( FILE *fid, Tvec_sp_ptr x )
{
    int i;
    
    ASSERT_MSG( x->init, "vec_sp_print() : sparse vector is uninitialised" );
    
    if( !x->block )
    {
        fprintf( fid, "sparse vector of length %d with %d nz elements\n", x->n, x->nnz );
        for( i=0; i<x->nnz; i++ )
            fprintf( fid, "[%d, %g]  ", x->indx[i], x->nz[i] );
    }
    else
    {
        int j;
        
        fprintf( fid, "sparse block vector of length %d with %d nz elements\n", x->n, x->nnz );
        for( i=0; i<(x->nnz BLOCK_V_SHIFT); i+=(1 BLOCK_V_SHIFT) )
        {
            fprintf( fid, "\t(%d)\t", x->indx[i/(1 BLOCK_V_SHIFT)] );
            for( j=i; j<i+(1 BLOCK_V_SHIFT); j++ )
            {
                fprintf( fid, "%g\t", x->nz[j] );
            }
            fprintf( fid, "\n" );
        }
    }
    
    fprintf( fid, "\n" );
}

void vec_sp_free( Tvec_sp_ptr x )
{
    ASSERT_MSG( x->init, "vec_sp_free() : attempt to free an uninitialised sparse vector" );
    
    free( x->nz );
    free( x->indx );
    x->init = 0;
    x->block = 0;
}

/*
    copy a vector from one location to another
 
    y = x
*/
void vec_sp_copy( Tvec_sp_ptr x, Tvec_sp_ptr y )
{
    int i;
    
    ASSERT_MSG( x->init, "vec_sp_copy() : source vector must be initialised" );
    if( y->init )
    {
        vec_sp_free( y );
    }
    vec_sp_init( y, x->n, x->nnz, x->block );
    
    // copy over the nz elements
    if( !x->block )
        dcopy( x->nnz, x->nz, 1, y->nz, 1 );
    else
        dcopy( (x->nnz BLOCK_V_SHIFT), x->nz, 1, y->nz, 1 );
    
    // copy over the indices
    for( i=0; i<x->nnz; i++ )
    {
        y->indx[i] = x->indx[i];
    }
}

/*
    perform a sparse-sparse innerproduct
*/
double vec_sp_ip( Tvec_sp_ptr x, Tvec_sp_ptr y )
{
    int xi, yi;
    double ip;
    
    xi = yi = 0;
    ip = 0.;
    
    // not block
    if( !x->block )
    {
        while( xi<x->nnz && yi<y->nnz  )
        {
            while( (yi<y->nnz) && (x->indx[xi] > y->indx[yi]) )
            {
                yi++;
            }
            if( (yi<y->nnz) && (x->indx[xi] == y->indx[yi]) )
            {
                ip += x->nz[xi]*y->nz[yi];
                yi++;
            }
            xi++;
        }
    }
    // block storage
    else
    {
        double *xp, *yp;
        
        while( xi<x->nnz && yi<y->nnz  )
        {
            while( (yi<y->nnz) && (x->indx[xi] > y->indx[yi]) )
            {
                yi++;
            }
            if( (yi<y->nnz) && (x->indx[xi] == y->indx[yi]) )
            {
                xp = x->nz + (xi BLOCK_V_SHIFT);
                yp = y->nz + (yi BLOCK_V_SHIFT);
                ip += BLOCK_V_IP( xp, yp );
                yi++;
            }
            xi++;
        }
    }
    return ip;
}

/*
    find the 2 norm of a sparse vector
*/
double vec_sp_nrm2( Tvec_sp_ptr x )
{
    ASSERT_MSG( x->init, "vec_sp_2norm() : vector must be initialised" );
    
    if( !x->block )
        return dnrm2( x->nnz, x->nz, 1);
    else
        return dnrm2( (x->nnz BLOCK_V_SHIFT), x->nz, 1); 
}

/*
    expand the sparse vector in x to a dense vector in y
*/
void vec_sp_scatter( Tvec_sp_ptr x, double *y )
{
    int i;
    
    if( !x->block )
    {
        // zero out the destination
        for( i=0; i<x->n; i++ )
            y[i] = 0.;
        
        // scatter the nz elements
        for( i=0; i<x->nnz; i++ )
            y[x->indx[i]] = x->nz[i];
    }
    else
    {       
        // zero out the destination
        for( i=0; i<(x->n BLOCK_V_SHIFT); i++ )
            y[i] = 0.;
        
        // scatter the nz elements
        for( i=0; i<x->nnz; i++ )
        {
            y[x->indx[i] BLOCK_V_SHIFT] = x->nz[i BLOCK_V_SHIFT];
        }
    }
}

/*
    gather a dense vector into a sparse storage format, with a maximum of lfill
    elements, that is the lfill elements of largest absolute value are selected
 
    y = sorted x values, return value is the number of values found
 
    the permutation vector holds the permutation required for the reorder
 */
int dense_gather_lfill( double *x, double *y, int *p, int n, int lfill )
{
    int i, pos;
    
    // set up the permutation vector
    for( i=0; i<n; i++ )
    {
        p[i] = i;
    }
    
    // sort the elements in order of absolute value
    // keeping a permutation vector at the same time
    heapsort_double_index( n, p, x );
    
    // remove any zero values from the lfill largest elements
    for( i=n-lfill, pos=0; i<n; i++, pos++ )
    {
        y[pos] = x[p[i]];
        p[pos] = p[i];
    }
    
    // just return lfill for the time being
    return lfill;
}

/*
    perform a sparse-sparse matrix vector multiply
    y = Ax
 
    y is dense format
 */
void mtx_CRS_gemv_ssd( Tmtx_CRS_ptr A, Tvec_sp_ptr x, double *y )
{
    double *xd;
    int n, m;
    
    n = A->nrows;
    m = A->ncols;
    
    // test that valid arguments have been passed
#ifdef DEBUG
    ASSERT_MSG( A->init && x->init, "mtx_CRS_vec_sp_mul() : A and x must be initialised" );
    ASSERT_MSG( A->ncols == x->n, "mtx_CRS_sp_mul() : matrix and vector dimensions do not match" );
    ASSERT_MSG( (A->block && x->block) || (!A->block && !x->block) , "mtx_CRS_sp_mul() : both matrix and vector must be block/scalar");
#endif
    
    // initialise variables     
    if( !x->block )
    {
        xd = (double *)malloc( sizeof(double)*m ); // memory for dense version of x
        y = (double*)malloc( n*sizeof(double) );
    }
    else
    {
        xd = (double *)malloc( sizeof(double)*(m BLOCK_V_SHIFT) );
        y = (double*)malloc( (n BLOCK_V_SHIFT)*sizeof(double) );
    }
    vec_sp_scatter( x, xd );  // put x into dense form
    
    // perform the multiplication
    mtx_CRS_vec_mul( A, xd, y );
    
    // free up memory
    free( xd );
}

/*
    perform a sparse-sparse matrix vector multiply
    y = Ax
*/
void mtx_CRS_gemv_sss( Tmtx_CRS_ptr A, Tvec_sp_ptr x, Tvec_sp_ptr y )
{
    double *xd, *yd;
    int n, m;
    
    n = A->nrows;
    m = A->ncols;
    
    // test that valid arguments have been passed
#ifdef DEBUG
    ASSERT_MSG( A->init && x->init, "mtx_CRS_vec_sp_mul() : A and x must be initialised" );
    ASSERT_MSG( A->ncols == x->n, "mtx_CRS_sp_mul() : matrix and vector dimensions do not match" );
    ASSERT_MSG( (A->block && x->block) || (!A->block && !x->block) , "mtx_CRS_sp_mul() : both matrix and vector must be block/scalar");
#endif
    
    // initialise variables
    vec_sp_init( y, n, n, x->block ); // memory for solution
    if( !x->block )
    {
        xd = (double *)malloc( sizeof(double)*m ); // memory for dense version of x
        yd = (double *)malloc( sizeof(double)*n );
    }
    else
    {
        xd = (double *)malloc( sizeof(double)*(m BLOCK_V_SHIFT) );
        yd = (double *)malloc( sizeof(double)*(n BLOCK_V_SHIFT) );
    }
    vec_sp_scatter( x, xd );  // put x into dense form
    
    // perform the multiplication
    mtx_CRS_vec_mul( A, xd, yd );
    
    // save y in sparse form    
    vec_sp_gather( y, yd, n, A->block );

    // free up memory
    free( xd );
    free( yd );
}

/*
    perform a sparse-dense matrix vector multiply
 
    works for both scalar and block entries.
 
    y = Ax
*/
void mtx_CRS_vec_mul( Tmtx_CRS_ptr A, double *x, double *y )
{
    int row, i;
    
    ASSERT_MSG( A->init , "mtx_CRS_vec_mul() : A must be initialised" );
    
    // not a block matrix
    if( !A->block )
    {
        // initialise the output vector to zero
        for( row=0; row<A->nrows; row++ )
        {
            y[row] = 0.;
        }
        
        // perform the multiplication
        for( row=0; row<A->nrows; row++ )
        {
            for( i=A->rindx[row]; i<A->rindx[row+1]; i++ )
            {
                y[row] += A->nz[i] + x[A->cindx[i]];
            }
        }
    }
    // block matrix
    else
    {
        double *posm, *posy, *posx;
        
        // initialise the output vector to zero
        for( row=0; row<(A->nrows BLOCK_V_SHIFT); row++ )
        {
            y[row] = 0.;
        }
        
        // perform the multiplication
        posy = y;
        posm = A->nz;
        for( row=0; row<A->nrows; row++, posy+=(1 BLOCK_V_SHIFT))
        {
            for( i=A->rindx[row]; i<A->rindx[row+1]; i++, posm+=(1 BLOCK_M_SHIFT) )
            {
                posx = x + (A->cindx[i] BLOCK_M_SHIFT);
                BLOCK_MV_MULT( posy, posm, posx );
            }
        }
    }
    
}

/*************************************************
 *      mtx_CRS_print()
 *
 *      print out a CRS matrix sparsity
 *************************************************/
void mtx_CRS_print( FILE *fid, Tmtx_CRS_ptr A )
{
    int i;
    
    ASSERT_MSG( A->init , "mtx_CRS_print() : A must be initialised" );
    
    fprintf( fid, "sparse CRS matrix has dimensions %dX%d with %d nz elements\n", A->nrows, A->ncols, A->nnz );
    fprintf( fid, "rindx\n" );
    for( i=0; i<A->nrows+1; i++ )
        fprintf( fid, "\t(%d)\t%d\n", i, A->rindx[i] );
    fprintf( fid, "cindx\n" );
    for( i=0; i<A->nnz; i++ )
        fprintf( fid, "\t(%d)\t%d\n", i, A->cindx[i] );
    fprintf( fid, "nz\n" );
    for( i=0; i<A->nnz; i++ )
        fprintf( fid, "\t(%d)\t%g\n", i, A->nz[i] );
}

/*************************************************
*       mtx_CCS_print()
*
*       print out a CCS matrix sparsity
*************************************************/
void mtx_CCS_print( FILE *fid, Tmtx_CCS_ptr A )
{
    int i;
    
    ASSERT_MSG( A->init , "mtx_CCS_print() : A must be initialised" );
    
    fprintf( fid, "sparse CCS matrix has dimensions %dX%d with %d nz elements\n", A->nrows, A->ncols, A->nnz );
    fprintf( fid, "cindx\n" );
    for( i=0; i<A->ncols+1; i++ )
        fprintf( fid, "\t(%d)\t%d\n", i, A->cindx[i] );
    fprintf( fid, "rindx\n" );
    for( i=0; i<A->nnz; i++ )
        fprintf( fid, "\t(%d)\t%d\n", i, A->rindx[i] );
    fprintf( fid, "nz\n" );
    for( i=0; i<A->nnz; i++ )
        fprintf( fid, "\t(%d)\t%g\n", i, A->nz[i] );
}

/**************************************************
 trans = 'N' : y = alpha*A*x + beta*y
 trans = 'T' : y = alpha*A'*x + beta*y
*************************************************/
void mtx_CRS_gemv( Tmtx_CRS_ptr A, double *x, double *y, double alpha, double beta, char trans )
{
    int row, col;
    
    ASSERT_MSG( A->init, "mtx_CRS_gemv() : matrix must be initialised" );
    ASSERT_MSG( (trans=='N' || trans=='n' || trans=='T' || trans=='t'), "mtx_CRS_gemv() : invalid trans value" );
    
    if( trans=='N' || trans=='n' )
    {   
        if( !A->block )
        {
            for( row=0; row<A->nrows; row++ )
            {
                y[row] *= beta;
                for( col=A->rindx[row]; col<A->rindx[row+1]; col++ )
                {
                    y[row] += alpha*A->nz[col]*x[A->cindx[col]];
                }
            }
        }
        else
        {
            double *xp, *yp, *Ap;
            
            yp = y;
            for( row=0; row<A->nrows; row++, yp+=(1 BLOCK_V_SHIFT) )
            {
                BLOCK_V_SCALE( beta, yp );
                for( col=A->rindx[row]; col<A->rindx[row+1]; col++ )
                {
                    Ap = A->nz + (col BLOCK_M_SHIFT);
                    xp = x + (A->cindx[col] BLOCK_V_SHIFT);
                    BLOCK_MV_AMULT( yp, Ap, xp, alpha );
                }
            }
        }
    }
    else
    {
        if( !A->block )
        {
            for( col=0; col<A->ncols; col++ )
            {
                y[col] *= beta;
            }               
            for( row=0; row<A->nrows; row++ )
            {
                for( col=A->rindx[row]; col<A->rindx[row+1]; col++ )
                {
                    y[A->cindx[col]] += alpha*A->nz[A->cindx[col]]*x[row];
                }
            }
        }
        else
        {
            double *xp, *yp, *Ap;
            
            yp = y;
            for( col=0; col<A->ncols; col++, yp+=(1 BLOCK_V_SHIFT) )
            {
                BLOCK_V_SCALE( beta, yp );
            }
            xp = x;
            for( row=0; row<A->nrows; row++, xp+=(1 BLOCK_V_SHIFT) )
            {
                
                for( col=A->rindx[row]; col<A->rindx[row+1]; col++ )
                {
                    Ap = A->nz + (A->cindx[col] BLOCK_M_SHIFT);
                    yp = y + (A->cindx[col] BLOCK_V_SHIFT);
                    BLOCK_MV_AMULT( yp, Ap, xp, alpha );
                }
            }
        }
    }
}

/*
    very simple, just find y = A*x
 
    if( yzero != 0 )
        y = 0*y + A*x;
    else
        y = y + A*x
 
    no other parameters just for speed
*/
void mtx_CCS_mat_vec_mult( Tmtx_CCS_ptr A, double *x, double *y, int yzero )
{
    int i, j;
    double val;

#ifdef DEBUG
    ASSERT_MSG( x!=NULL && y!=NULL, "mtx_CCS_mat_vec_mult() : x and y vectors must be initialised." );
    ASSERT_MSG( A->init, "mtx_CCS_mat_vec_mult() : A matrix must be initialised." );
#endif

    // zero out the y vector if requested
    if( yzero )
    {
        for( i=0; i<A->ncols; i++ )
        {
            y[i] = 0.;
        }
    }
    
    // loop over each column
    for( i=0; i<A->ncols; i++)
    {
        // only need to loop over This column if x[i] is nonzero
        if( x[i] )
        {
            val = x[i];
            for( j=A->cindx[i]; j<A->cindx[i+1]; j++ )
            {
                y[A->rindx[j]] += A->nz[j] * val;
            }
        }
    }
}

/******************************************************************
*       mtx_CCS_is_sym_structure()
*
*       tests if A has a symmetric sparsity pattern
*           returns 1 : symmetric
*           returns 0 : non-symmetric
******************************************************************/
int mtx_CCS_is_sym_structure( Tmtx_CCS_ptr A )
{
    long long *w, *w_transpose;
    long long top;
    int i, col, symm, pos;
    
    /*
     *      Trivial cases
     */
    
    // return 0 if not initialised
    if( !A->init )
        return 0;
    
    // return 0 if not square
    if( A->nrows!=A->ncols )
        return 0;
    
    /*
     *      find then sort the nz weights for CRS form
     */
    
    // find
    w           = (long long*)malloc( sizeof(long long)*A->nnz );
    w_transpose = (long long*)malloc( sizeof(long long)*A->nnz );
    pos = 0;
    for( col=0; col<A->ncols; col++ )
    {
        for( i=A->cindx[col]; i<A->cindx[col+1]; i++, pos++ )
        {
            w_transpose[pos] = (long long)A->rindx[i]*(long long)A->ncols + (long long)col;
        }
    }
    
    // sort
    heapsort_longlong( A->nnz, w_transpose );
    
    /*
     *      find the nz weights for the CCS form
     */
    
    pos = 0;
    top = 0;
    for( col=0; col<A->ncols; col++ )
    {
        for( i=A->cindx[col]; i<A->cindx[col+1]; i++, pos++ )
        {
            w[pos] = (long long)A->rindx[i] + (long long)top;
        }
        top += A->nrows;
    }  
    
    /*
     *      compare the weights
     */
    
    symm = 1;
    for( i=0; i<A->nnz; i++ )
    {
        if( w[i] != w_transpose[i] )
        {
            printf( "w[%d] != w_transpose[%d]\t\t%g != %g\n", i, i, (double)w[i], (double)w_transpose[i]  );
            symm = 0;
            break;
        }
    }
    
    /*
     *      clean up and go home
     */
    free( w );
    free( w_transpose );
    
    return symm;
}

/******************************************************************
*       mtx_CCS_is_diag_nz()
*
*       tests if A has nz entries on diagonal
*           returns 1 : all diagonal entries are nz
*           returns 0 : there are zero nz entries on diagonal
******************************************************************/
int mtx_CCS_is_diag_nz( Tmtx_CCS_ptr A )
{
    int col, i, is_nz;

    is_nz = 1;
    for( col=0; col<A->ncols && is_nz; col++ )
    {
        is_nz = 0;      
        for( i=A->cindx[col]; i<A->cindx[col+1]; i++ )
        {
            if( A->rindx[i]==col )
            {
                is_nz = 1;
                break;
            }
        }
    }
    
    if( !is_nz )
        printf( "\t\tdiagonal on column %d of %d is zero\n", col, A->ncols-1 );
        
    return is_nz;
}

/******************************************************************
*       mtx_CCS_is_connected()
*
*       tests if every node in the graph represented by A is
*       connected to at least one other node
******************************************************************/
int mtx_CCS_is_connected( Tmtx_CCS_ptr A )
{
    int col, is_conn=1, min, max, nnz;
    
    max = 0;
    min = A->nnz;
    for( col=0; col<A->ncols; col++ )
    {
        nnz = A->cindx[col+1]-A->cindx[col];
        if( nnz<min )
            min = nnz;
        if( nnz>max )
            max = nnz;
    }
    
    if( min<2 )
        is_conn=0;
    else
        is_conn=1;
    
    printf( "\t\tbetween %d -> %d nz per column\n", min, max );
    
    return is_conn;
}

/************************************************************************** 
*       mtx_CCS_validate()
*
*       check that a CCS matrix is self consistent
**************************************************************************/
int mtx_CCS_validate( Tmtx_CCS_ptr A )
{
    int col, val;
    
    /*
     *      Trivial cases
     */
    
    if( !A->init || !A->nrows || !A->ncols || !A->nnz )
        return 1;
    
    if( A->nrows<0 || A->ncols<0 || A->nnz<0 )
    {
        fprintf( stderr, "WARNING : mtx_CCS_validate() : invalid dimensions : %dX%d with %d nz\n", A->nrows, A->ncols, A->nnz );
        return 0;
    }
    
    if( A==NULL )
    {
        fprintf( stderr, "WARNING : mtx_CCS_validate() : passed NULL pointer to matrix\n" );
        return 0;
    }
    
    if( A->nrows*A->ncols < A->nnz )
    {
        fprintf( stderr, "WARNING : mtx_CCS_validate() : nnz=%d is too many for a  matrix of dimensions %dX%d\n", A->nnz, A->nrows, A->ncols );
        return 0;
    }
    
    /*
     *      check that cindx values are reasonable
     */
    if( A->cindx[0]!=0 )
    {
        fprintf( stderr, "WARNING : mtx_CCS_validate() : A->cindx[0] = %d != 0\n", A->cindx[0] );
        return 0;
    }
    for( col=0; col<A->ncols; col++ )
    {
        val = A->cindx[col+1]-A->cindx[col];
        if( val<0 || val>A->nrows )
        {
            fprintf( stderr, "WARNING : mtx_CCS_validate() : col %d cindx values are not consitent [%d->%d]\n", col, A->cindx[col], A->cindx[col+1] );
            return 0;
        }
    }
    if( A->cindx[col]!=A->nnz )
    {
        fprintf( stderr, "WARNING : mtx_CCS_validate() : cindx[ncols] = %d != nnz = %d\n", A->cindx[col], A->nnz );
        return 0;
    }
    
    /*
     *      check that the rindx values are reasonable
     */
    for( col=0; col<A->ncols; col++ )
    {
        val = A->cindx[col];
        if( A->rindx[val]<0 || A->rindx[val]>=A->nrows )
        {
            fprintf( stderr, "WARNING : mtx_CCS_validate() : rindx value = %d (0-%d) out of range\n", A->rindx[val], A->nrows-1 );
            return 0;
        }
        for( ++val; val<A->cindx[col+1]; val++ )
        {
            if( A->rindx[val-1]>=A->rindx[val] )
            {
                fprintf( stderr, "WARNING : mtx_CCS_validate() : rindx values not in order : rows [%d, %d], col %d, indx [%d %d]\n", A->rindx[val-1], A->rindx[val], col, val-1, val );
                return 0;
            }
            if( A->rindx[val]<0 || A->rindx[val]>=A->nrows )
            {
                fprintf( stderr, "WARNING : mtx_CCS_validate() : rindx value = %d (0-%d) out of range\n", A->rindx[val], A->nrows-1 );
                return 0;
            }
        }
    }
    
    /*
     *      If we got This far then the matrix is A-OK
     */
    
    return 1;
}

/************************************************************************** 
*       mtx_CRS_validate()
*
*       check that a CRS matrix is self consistent
**************************************************************************/
int mtx_CRS_validate( Tmtx_CRS_ptr A )
{
    int row, val;
    
    /*
     *      Trivial cases
     */
    
    if( !A->init || !A->nrows || !A->ncols || !A->nnz )
        return 1;
    
    if( A->nrows<0 || A->ncols<0 || A->nnz<0 )
    {
        sWARNING( sprintf( _errmsg, "mtx_CRS_validate() : invalid dimensions : %dX%d with %d nz\n", A->nrows, A->ncols, A->nnz ));
        return 0;
    }
    
    if( A==NULL )
    {
        sWARNING( sprintf( _errmsg, "mtx_CRS_validate() : passed NULL pointer to matrix\n" ));
        return 0;
    }
    
    if( A->nrows*A->ncols < A->nnz )
    {
        sWARNING( sprintf( _errmsg, "mtx_CRS_validate() : nnz=%d is too many for a  matrix of dimensions %dX%d\n", A->nnz, A->nrows, A->ncols ));
        return 0;
    }
    
    /*
     *      check that rindx values are reasonable
     */
    if( A->rindx[0]!=0 )
    {
        sWARNING( sprintf( _errmsg, "mtx_CRS_validate() : A->rindx[0] = %d != 0\n", A->rindx[0] ));
        return 0;
    }
    for( row=0; row<A->nrows; row++ )
    {
        val = A->rindx[row+1]-A->rindx[row];
        if( val<0 || val>A->ncols )
        {
            sWARNING( sprintf( _errmsg, "mtx_CRS_validate() : row %d rindx values are not consitent [%d->%d]\n", row, A->rindx[row], A->rindx[row+1] ));
            return 0;
        }
    }
    if( A->rindx[row]!=A->nnz )
    {
        sWARNING( sprintf( _errmsg, "mtx_CRS_validate() : rindx[nrows] = %d != nnz = %d", A->rindx[row], A->nnz ));
        return 0;
    }
    
    /*
     *      check that the cindx values are reasonable
     */
    for( row=0; row<A->nrows; row++ )
    {
        val = A->rindx[row];
        if( A->cindx[val]<0 || A->cindx[val]>=A->ncols )
        {
            sWARNING( sprintf( _errmsg, "mtx_CRS_validate() : cindx value = %d (0-%d) out of range\n", A->cindx[val], A->ncols-1 ));
            return 0;
        }
        for( ++val; val<A->rindx[row+1]; val++ )
        {
            if( A->cindx[val-1]>=A->cindx[val] )
            {
                sWARNING( sprintf( _errmsg, "mtx_CRS_validate() : cindx values not in order : cindx[%d]=%d, cindx[%d]=%d, on row %d\n", val-1, A->cindx[val-1], val, A->cindx[val], row ));
                return 0;
            }
            if( A->cindx[val]<0 || A->cindx[val]>=A->ncols )
            {
                sWARNING( sprintf( _errmsg, "mtx_CRS_validate() : cindx value = %d (0-%d) out of range\n", A->cindx[val], A->ncols-1 ));
                return 0;
            }
        }
    }
    
    /*
     *      If we got This far then the matrix is A-OK
     */
    
    return 1;
}

/*
 boolean test if matrix A==B
 
 if( A==B )
 return 1
 else
 return 0
 */
int mtx_CRS_equal( Tmtx_CRS_ptr A, Tmtx_CRS_ptr B )
{
    int i;
    
    /*
     *  simple cases
     */
    if( !A->init || !B->init )
        return 0;
    
    if( A->nnz!=B->nnz )
        return 0;
    
    if( (A->nrows!=B->nrows) || (A->ncols!=B->ncols) )
        return 0;
    
    /*
     *  row indices
     */
    for( i=0; i<A->nrows+1; i++ )
        if( A->rindx[i]!=B->rindx[i] )
            return 0;
            
    if( A->block )
    {
        for( i=0; i<A->nnz; i++ )
            if( (A->cindx[i]!=B->cindx[i])  || (A->nz[i]!=B->nz[i]) )
                return 0;
    }
    else
    {
        for( i=0; i<A->nnz; i++ )
        {
            if( (A->cindx[i]!=B->cindx[i]) )
                return 0;
        }
        for( i=0; i<(A->nnz BLOCK_M_SHIFT); i++ )
        {
            if( (A->nz[i]!=B->nz[i]) )
                return 0;
        }       
    }
    return 1;
}

/*
 unpack a column of a block matrix into a vector. The block column is BLOCK_SIZE columns of
 the non-block matrix. the columns are stored end-to-end in v
 
 A is the matrix from which to get the column 
 v is the vector in which to unbpack the column (must be initialised by caller)
 col is the column index (block numbering) of the column to unpack
*/
void mtx_CCS_column_unpack_block( Tmtx_CCS_ptr A, double *v, int col )
{
    int i, pos, cindx;
    double *copy_from, *copy_to;
    
    ASSERT_MSG( A->block, "mtx_CCS_column_unpack_block() : the matrix must be block format" );
    ASSERT_MSG( col>=0 && col<A->ncols, "mtx_CCS_column_unpack_block() : the column requested is out of range" );
    
    // unpack the BLOCK_SIZE columns one at a time
    cindx = A->cindx[col];
    for( i=0; i<(A->nrows BLOCK_M_SHIFT); i++ )
        v[i] = 0.;
    for( i=0; i<BLOCK_SIZE; i++ )
    {
        copy_from = A->nz + (cindx BLOCK_M_SHIFT) + (i BLOCK_V_SHIFT);
        for( pos=A->cindx[col]; pos<A->cindx[col+1]; pos++, copy_from+=(1 BLOCK_M_SHIFT) )
        {
            copy_to = v + (A->rindx[pos] BLOCK_V_SHIFT);
            BLOCK_V_COPY( copy_from, copy_to );
        }       
        v += (A->nrows BLOCK_V_SHIFT);
    }
}

/*
 add an a column from a block matrix A (width BLOCK_SIZE scalar columns) to an ubpacked
 set of scalar columns u.
 
 u = alpha*A + beta*u
 */
void mtx_CCS_column_add_unpacked_block( Tmtx_CCS_ptr A, double *u, int col, double alpha, double beta )
{
    int i, pos, cindx;
    double *copy_from, *copy_to;
    
    ASSERT_MSG( A->block, "mtx_CCS_column_unpack_block() : the matrix must be block format" );
    ASSERT_MSG( col>=0 && col<A->ncols, "mtx_CCS_column_unpack_block() : the column requested is out of range" );
    
    // add the BLOCK_SIZE columns one at a time
    cindx = A->cindx[col];
    for( i=0; i<BLOCK_SIZE; i++ )
    {
        copy_from = A->nz + (cindx BLOCK_M_SHIFT) + (i BLOCK_V_SHIFT);
        for( pos=A->cindx[col]; pos<A->cindx[col+1]; pos++, copy_from+=(1 BLOCK_M_SHIFT) )
        {
            copy_to = u + (A->rindx[pos] BLOCK_V_SHIFT);
            BLOCK_V_AXBU( copy_to, copy_from, alpha, beta );
        }       
        u += (A->nrows BLOCK_V_SHIFT);
    }
}

/************************************************************
 *  mtx_CRS_col_normalise()
 *
 *  normalise the columns of a CRS matrix
 *  For block matrices the columns are normalised as if it were
 *  a scalar matrix
 *
 *  on return the vector norms contains the norm of each column,
 *  memory for norms is allocated by the caller, and norms
 *  is of length legth A.ncols for scalar and A.ncols*BLOCK_SIZE
 *  for block matrices
 ************************************************************/
void mtx_CRS_col_normalise( Tmtx_CRS_ptr A, double *norms )
{
    if( !A->init )
    {
        fprintf( stderr, "ERROR : mtx_CRS_col_normalise() : passed an uninitialised matrix\n" );
        exit(1);
    }
    mtx_CRS_col_norms( A, norms );
    mtx_CRS_col_scale( A, norms, 0 );
}

/************************************************************
*   mtx_CRS_col_norms()
*
*   fine the column norms of a CRS matrix
*   For block matrices the columns are treated as if it they
*   a scalar matrix
*
*   on return the vector norms contains the norm of each column,
*   memory for norms is allocated by the caller, and norms
*   is of length legth A.ncols for scalar and A.ncols*BLOCK_SIZE
*   for block matrices
************************************************************/
void mtx_CRS_col_norms( Tmtx_CRS_ptr A, double *norms )
{
    int i, ncol, block, pos;
    double *A_pos, *norm_pos;
    
    // initialise parameters
    block = A->block;
    ncol = A->ncols;
    if( block )
        ncol = (ncol BLOCK_V_SHIFT);
    
    // set norms to be zeros
    for( i=0; i<ncol; i++ )
        norms[i] = 0.;
    
    // find the norm of each column
    if( block )
    {
        for( i=0, pos=0; i<A->nnz; i++, pos+=BLOCK_SIZE*BLOCK_SIZE )
        {
            norm_pos = norms + (A->cindx[i] BLOCK_V_SHIFT);
            A_pos = A->nz + pos;
            BLOCK_M_COLSUM_SQUARES_CUM( A_pos, norm_pos );
        }
    }
    else
        for( i=0; i<A->nnz; i++ )
            norms[A->cindx[i]] += A->nz[i]*A->nz[i];
    for( i=0; i<ncol; i++ )
        norms[i] = sqrt( norms[i] );
}

/************************************************************
*   mtx_CRS_col_scale()
*
*   scale the columns of a CRS matrix
*   For block matrices the columns are treated as if it they
*   a scalar matrix
*
*   the scale factor for each column is in the vectors scales
*   op = 1 : each column is scaled by multiplication
*   op = 0 : each column is scaled by division
************************************************************/
void mtx_CRS_col_scale( Tmtx_CRS_ptr A, double *scales, int op )
{
    int i, block, pos;
    double *A_pos, *scale_pos;
    
    // initialise parameters
    block = A->block;
    
    // scale each column by multiplication
    if( op )
    {
        if( block )
        {
            for( i=0, pos=0; i<A->nnz; i++, pos+=BLOCK_SIZE*BLOCK_SIZE )
            {
                scale_pos = scales + (A->cindx[i] BLOCK_V_SHIFT);
                A_pos = A->nz + pos;
                BLOCK_M_COLSCALE_MUL( A_pos, scale_pos );
            }
        }
        else
            for( i=0; i<A->nnz; i++ )
                A->nz[i] *= scales[A->cindx[i]];
    }
    else
    // scale each column by division
    {
        if( block )
        {
            for( i=0, pos=0; i<A->nnz; i++, pos+=BLOCK_SIZE*BLOCK_SIZE )
            {
                scale_pos = scales + (A->cindx[i] BLOCK_V_SHIFT);
                A_pos = A->nz + pos;
                BLOCK_M_COLSCALE_DIV( A_pos, scale_pos );
            }
        }
        else
            for( i=0; i<A->nnz; i++ )
                A->nz[i] /= scales[A->cindx[i]];
    }
}

