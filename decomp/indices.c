/*
 *  indices.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "indices.h"
#include "benlib.h"

void index_init( Tindex_ptr index, int dim_major, int dim_minor )
{
	if( dim_major )
	{
		index->dim_major = dim_major;
		index->dim_minor = dim_minor; 
		index->index_major = (int *)malloc( sizeof(int)*dim_major );
		index->index_minor = (int *)malloc( sizeof(int)*(dim_minor+1) );
		index->index_minor[0] = 0;
	}
	else
	{
		index->dim_major   = index->dim_minor   = 0;
		index->index_major = index->index_minor = NULL; 
	}
}

void index_free( Tindex_ptr index )
{
	index->dim_major = index->dim_minor = 0;
	if( index->index_major )
		free( index->index_major );
	if( index->index_minor )
		free( index->index_minor );
	index->index_major = index->index_minor = NULL;
}

void index_split( int *index_major, int *index_minor, int *index_dist, int dim_major, int dim_minor, int n_dom, Tindex_ptr split, int normalise )
{
	int i, j, pos=0, n_el, end;
	int *dims, **splits;
	Tindex_ptr index;
	
	// allocate memory
	dims = (int *)calloc( n_dom, sizeof(int) );
	splits = (int **)malloc( sizeof(int*)*dim_minor );
	for( i=0; i<dim_minor; i++ )
		splits[i] = (int *)malloc( sizeof(int)*(n_dom+1) );
	
	// loop through the minor axis, recording where the splits are
	for( j=0; j<dim_minor; j++ )
	{
		pos = index_minor[j];
		end = index_minor[j+1];
		splits[j][0] = pos;
		
		// determine where the domain splits lie on This row/column lie
		for( i=0; i<n_dom; i++ )
		{
			while( pos<end && index_major[pos]<index_dist[i+1] )
				pos++;
			splits[j][i+1] = pos;
			dims[i] += (pos-splits[j][i]);
		}		
	}
	// a little bit of code to check that the caller isn't a goose
	ASSERT_MSG( pos==dim_major, "index_split() : the indices you passed are pear-shaped." );
	
	// allocate memory for the split indices
	for( i=0; i<n_dom; i++ )
	{
		index_init( split + i, dims[i], dim_minor );
	}
	
	// split the indices, looping over the domains
	for( i=0; i<n_dom; i++ )
	{
		// point to the index for domain i
		index = split + i;
		
		// create index list for This domain
		if( dims[i] )
		{
			pos = 0;
			for( j=0; j<dim_minor; j++ )
			{
				n_el = splits[j][i+1] - splits[j][i];
				if( n_el )
				{
					memcpy( index->index_major + pos, index_major + splits[j][i], n_el*sizeof(int) );
				}
				pos += n_el;
				index->index_minor[j+1] = pos;
			}
		}
	}
	
	// normalise all of the split indices to be local if needed
	if( normalise )
	{
		// do it one domain at a time
		for( i=0; i<n_dom; i++ )
		{
			if( dims[i] )
			{
				index = split + i;
				j = index_dist[i];
				for( pos=0; pos<dims[i]; pos++ )
				{
					index->index_major[pos] -= j;
				}
			}
		}
	}
	
	// free memory
	free( dims );
	for( i=0; i<dim_minor; i++ )
		free( splits[i] );
	free( splits );	
}


/********************************************************************************************
 *		index_CRS_to_CCS()
 *
 *		convert a CRS storage to CCS storage
 ********************************************************************************************/
/*void index_CRS_to_CCS_( int *cindxCRS, int *rindxCRS, int *cindxCCS, int *rindxCCS, int *p, int nnz, int nrows, int ncols )
{
	int i, row, col;
	long long *w;
	long long j, pos;
	
	// initialise  the permutation array
	if( p!=NULL )
	{
		for( i=0; i<nnz; i++ )
			p[i] = i;
	}
		
	// calculate the column major weights for each nnz location
	w = (long long*)malloc( nnz*sizeof(long long) );
	for( row=0; row<nrows; row++ )
	{
		for( col=rindxCRS[row]; col<rindxCRS[row+1]; col++ )
		{
			w[ col ] = (long long)row + (long long)cindxCRS[col]*(long long)nrows;
		}
	}
	
	// sort the weights and keep a permutation vector
	if( p )
		heapsort_longlong_index( nnz, p, w );
	else
		heapsort_longlong( nnz, w );
	
	// calculate the CCS cindx and rindx values
	j   = (long long)nrows;
	pos = 0;
	row = 0;
	cindxCCS[0] = 0;
	for( col=0; col<ncols; col++ )
	{
		while( row<nnz && w[row]<j )
		{
			rindxCCS[row] = (int)(w[row] - pos);
			row++;
		}
		cindxCCS[col+1]=row;
		j   += (long long)nrows;
		pos += (long long)nrows;
	}
	
	// free up memory
	free( w );
}*/

void index_CRS_to_CCS( int *cindxCRS, int *rindxCRS, int *cindxCCS, int *rindxCCS, int *p, int nnz, int nrows, int ncols )
{
	int *counts;
	int i, j, row, col;
	
	counts = (int*)calloc( sizeof(int), ncols );
	
	for( i=0; i<nnz; i++ )
		counts[cindxCRS[i]]++;
	
	cindxCCS[0] = 0;
	for( i=0; i<ncols; i++ )
		cindxCCS[i+1] = cindxCCS[i] + counts[i];
	
	for( i=0; i<ncols; i++ )
		counts[i] = 0;
	
	if( p )
	{
		for( row=0; row<nrows; row++ )
		{
			for( i=rindxCRS[row]; i<rindxCRS[row+1]; i++ )
			{
				col = cindxCRS[i];
				j = cindxCCS[col] + counts[col];
				rindxCCS[j] = row;
				p[j] = i;
				counts[col]++;
			}
		}
	}
	else
	{
		for( row=0; row<nrows; row++ )
		{
			for( i=rindxCRS[row]; i<rindxCRS[row+1]; i++ )
			{
				col = cindxCRS[i];
				j = cindxCCS[col] + counts[col];
				rindxCCS[j] = row;
				counts[col]++;
			}
		}
	}
	
	free( counts );
}


/********************************************************************************************
	index_print()
 
	print out all information in index to the iostream stream.
********************************************************************************************/
void index_print( FILE *stream, Tindex_ptr index )
{
	int i;
	
	if( index->dim_major==0 )
	{
		fprintf( stream, "\tindex is empty.\n" );
		return;
	}
	
	fprintf( stream, "\tindex has dim_minor = %d, dim_major = %d\n\tindex_minor\n", index->dim_minor, index->dim_major );
	for( i=0; i<index->dim_minor+1; i++ )
	{
		fprintf( stream, "\t\t(%d)\t%d\n", i, index->index_minor[i] );
	}
	fprintf( stream, "\tindex_major\n" );
	for( i=0; i<index->dim_major; i++ )
	{
		fprintf( stream, "\t\t(%d)\t%d\n", i, index->index_major[i] );
	}
}

/********************************************************************************************
	index_cmp()

	compare two indices.
	returns 0 : equal
			1 : not equal - different dimensions
			2 : not equal - same dimensions, different values
********************************************************************************************/
int index_cmp( Tindex_ptr index1, Tindex_ptr index2 )
{
	int i;
	
	// both empty
	if( !index1->dim_major && !index2->index_major )
		return 0;
	
	// different dimensions
	if( (index1->dim_major != index2->dim_major) || (index1->dim_minor != index2->dim_minor) )
		return 1;
	
	for( i=0; i<index1->dim_minor+1; i++ )
		if( index1->index_minor[i]!=index2->index_minor[i] )
			return 2;
	
	for( i=0; i<index1->dim_major; i++ )
		if( index1->index_major[i]!=index2->index_major[i] )
			return 2;
	
	// return equality
	return 0;
}

/********************************************************************************************
	index_verify()

	verifys that an index contains realistic and consistant data.
	prints out a warning if the index is not valid

	returns 0 : valid
			1 : not valid
********************************************************************************************/
int index_verify( FILE *fid, Tindex_ptr index, int max_major )
{
	int i;
	
	if( !index->dim_major )
	{
		// empty list
		if( !index->dim_minor && !index->index_major && !index->index_minor )
		{
			return 0;
		}
		
		// invalid list
		fprintf( fid, "\tWARNING : index_verify() : dim_major==0 but not empty\n" );
		return 1;
	}
	
	// check that index_minor starts at 0
	if( index->index_minor[0]  )
	{
		fprintf( fid, "\tWARNING : index_verify() : index_minor[0]!=0\n" );
		return 1;
	}	
	
	// check that index_minor is in bounds 
	if( index->index_minor[index->dim_minor] != index->dim_major )
	{
		fprintf( fid, "\tWARNING : index_verify() : index_minor values out of range\n" );
		return 1;
	}
	
	// check that the dim_minor values are ascending
	for( i=0; i<index->dim_minor; i++ )
	{
		if( index->index_minor[i+1]<index->index_minor[i] )
		{
			fprintf( fid, "\tWARNING : index_verify() : index_minor values are not ascending [%d]>[%d]\n", i, i+1 );
			return 1;
		}
	}
	
	for( i=0; i<index->dim_major; i++ )
	{
		if( index->index_major[i]<0 || index->index_major[i]>=max_major )
		{
			fprintf( fid, "\tWARNING : index_verify() : index_major value out of bounds [%d]\n", i );
			return 1;
		}
	}
	
	return 0;
}

/********************************************************************************************
*		index_CRS2CCS_findperm()
*
*		find the permutation required when converting a CRS index to CCS
*
*		-   perm and order have been allocated length nnz
*		-   cindx and rindx have been initialised correctly
********************************************************************************************/
void index_CRS_to_CCS_findperm( int *cindx, int *rindx, int *perm, int *rindxCCS, long long *order, int nrows, int nnz )
{
	int pos, row;
	
	// initialise permutation array
	for( pos=0; pos<nnz; pos++ )
		perm[pos] = pos;
	
	// determine the colun major indices of nz values of E
	for( row=0; row<nrows; row++ )
		for( pos=rindx[row]; pos<rindx[row+1]; pos++)
			order[pos] = (long long)row + (long long)cindx[pos]*(long long)nrows;
	
	// sort the column major indices to get the permutation array
	heapsort_longlong_index( nnz, perm, order );
	
	// now sort out the CCS rindx values
	for( row=0; row<nrows; row++ )
		for( pos=rindx[row]; pos<rindx[row+1]; pos++ )
			rindxCCS[perm[pos]] = row;
}

/********************************************************************************************
*		index_make_adjacency()
*
*		form an adjacency graph for the CRS sparsity pattern cindx/rindx
*
*		output in _cindx/_rindx, with _cindx and _rindx allocated before
*		calling index_make_adjacency()
********************************************************************************************/
void index_make_adjacency( int n_nodes, int diag_shift, int *cindx, int *rindx, int *_cindx, int *_rindx )
{
	int k, i, pos;
	
	_rindx[0] = 0;
	pos = 0;
	
	for( i=0; i<n_nodes; i++ )
	{
		// cover up the diagonal element
		for( k=rindx[i]; k<rindx[i+1]; k++ )
			if( cindx[k] != (i+diag_shift) )
				_cindx[pos++] = cindx[k];
		
		// fix up the new rindx
		_rindx[i+1] = pos;
	}
}

