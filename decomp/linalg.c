#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "linalg_sparse.h"
#include "linalg_dense.h"
#include "benlib.h"

/******************************************************************************************
	converts a CCS matrix to dense format
	
	target matrix to must be intialised with correct dimensions, or uninitialised.
 
	it is your responsibility to ensure that from has been initialised properly, though the
	code does do some error checking.
 ******************************************************************************************/
void mtx_CCS_to_mtx( Tmtx_CCS_ptr from, Tmtx_ptr to )
{
	int i, j, col, shiftm, shiftv;
	double *src, *dest;

#ifdef DEBUG
	ASSERT_MSG( from->init, "mtx_CCS_to_mtx() : argument (Tmtx_CCS_ptr)from not initialised" );
	ASSERT_MSG( from->nrows>0 && from->ncols>0, "mtx_CCS_to_mtx() : argument (Tmtx_CCS_ptr)from is an empty matrix" );
#endif
	
	if( from->block )
	{
		double *pos;
		
		// allocate relevent memory for target matrix
		mtx_init( to, (from->nrows * BLOCK_SIZE), (from->ncols * BLOCK_SIZE) );
		
		shiftv = (from->nrows BLOCK_V_SHIFT);
		shiftm = (from->nrows BLOCK_M_SHIFT);
		pos = to->dat;
		src = from->nz;
		// now copy over the data, one column at a time
		for( col=0; col<from->ncols; col++, pos+=shiftm )
		{
			// copy over nz entries from This column
			for( i=from->cindx[col]; i<from->cindx[col+1]; i++ )
			{
				dest = pos + (from->rindx[i] BLOCK_V_SHIFT);
				for( j=0; j<BLOCK_SIZE; j++ )
				{
					BLOCK_V_COPY( src, dest );
					src += BLOCK_SIZE;
					dest += shiftv;		
				}
			}		
		}
	}
	else
	{
		int pos;
		
		// allocate relevent memory for target matrix
		mtx_init( to, from->nrows, from->ncols );
		
		// now copy over the data, one column at a time
		for( col=0, pos=0; col<from->ncols; col++, pos+=from->nrows )
		{
			// copy over nz entries from This column
			for( i=from->cindx[col]; i<from->cindx[col+1]; i++ )
			{
				to->dat[pos+from->rindx[i]] = from->nz[i];
			}		
		}
	}

}


/******************************************************************************************
	converts a CRS matrix to dense format
	
	target matrix to must be intialised with correct dimensions, or uninitialised.
 
	it is your responsibility to ensure that from has been initialised properly, though the
	code does do some error checking.
 ******************************************************************************************/
void mtx_CRS_to_mtx( Tmtx_CRS_ptr from, Tmtx_ptr to )
{
	int i, row, shift, shiftv, j;
	double *dest, *src;
	
#ifdef DEBUG
	ASSERT_MSG( from->init, "mtx_CRS_to_mtx() : argument (Tmtx_CRS_ptr)from not initialised" );
	if( !(from->nrows>0 && from->ncols>0) )
		printf( "Nrows %d \t ncols %d\n", from->nrows, from->ncols );
	ASSERT_MSG( from->nrows>0 && from->ncols>0, "mtx_CRS_to_mtx() : argument (Tmtx_CRS_ptr)from is an empty matrix" );
#endif
	
	if( from->block )
	{
		double *pos;
		
		// allocate relevent memory for target matrix
		mtx_init( to, (from->nrows * BLOCK_SIZE), (from->ncols * BLOCK_SIZE) );
		
		shift = from->nrows BLOCK_M_SHIFT;
		shiftv = from->nrows BLOCK_V_SHIFT;
		src = from->nz;
		pos = to->dat;
		
		// now copy over the data, one row at a time
		for( row=0; row<from->nrows; row++, pos+=BLOCK_SIZE )
		{
			// copy over nz entries from This column
			for( i=from->rindx[row]; i<from->rindx[row+1]; i++ )
			{
				/******************************************
				to->dat[pos+from->cindx[i]*from->nrows] = from->nz[i];
				******************************************/
				dest = pos + (from->cindx[i] * shift);
				for( j=0; j<BLOCK_SIZE; j++ )
				{
					BLOCK_V_COPY( src, dest );
					src += BLOCK_SIZE;
					dest += shiftv;		
				}
			}		
		}
	}
	else
	{
		int pos;
		
		// allocate relevent memory for target matrix
		mtx_init( to, from->nrows, from->ncols );
		
		// now copy over the data, one row at a time
		for( row=0, pos=0; row<from->nrows; row++, pos++ )
		{
			// copy over nz entries from This row
			for( i=from->rindx[row]; i<from->rindx[row+1]; i++ )
			{
				to->dat[pos+from->cindx[i]*from->nrows] = from->nz[i];
			}		
		}
	}
}
