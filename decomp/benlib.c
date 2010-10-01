/*
	benlib.c

	source file for Ben's collection of general purpose stuff
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/times.h>
#include <sys/resource.h>

#include "benlib.h"

// this has been updated from UQ

/*
 GLOBAL VARIABLE USED BY BLOCK matrix operations
 */

int _BLOCK_SIZE = BLOCK_SIZE;
int _LWORK = LWORK;
int _INFO;
int _IPIV[BLOCK_SIZE];
double _WORK[LWORK];


/*
	The following merge-sort code was downloaded of the site
		http://linux.wku.edu/~lamonml/algor/sort/merge.html
 
	Thanks to Michael Lamont for making This code publically available.
*/

/*
	Merge sort shell algorithm
*/
void mergeSort(int numbers[], int temp[], int array_size)
{
	m_sort(numbers, temp, 0, array_size - 1);
}

/*
	recursive part of the merge algorithm
*/
void m_sort(int numbers[], int temp[], int left, int right)
{
	int mid;
	
	if (right > left)
	{
		mid = (right + left) / 2;
		m_sort(numbers, temp, left, mid);
		m_sort(numbers, temp, mid+1, right);
		
		merge(numbers, temp, left, mid+1, right);
	}
}

/*
	does the actual merging of two arrays
*/
void merge(int numbers[], int temp[], int left, int mid, int right)
{
	int i, left_end, num_elements, tmp_pos;
	
	left_end = mid - 1;
	tmp_pos = left;
	num_elements = right - left + 1;
	
	while ((left <= left_end) && (mid <= right))
	{
		if (numbers[left] <= numbers[mid])
		{
			temp[tmp_pos] = numbers[left];
			tmp_pos = tmp_pos + 1;
			left = left +1;
		}
		else
		{
			temp[tmp_pos] = numbers[mid];
			tmp_pos = tmp_pos + 1;
			mid = mid + 1;
		}
	}
	
	while (left <= left_end)
	{
		temp[tmp_pos] = numbers[left];
		left = left + 1;
		tmp_pos = tmp_pos + 1;
	}
	while (mid <= right)
	{
		temp[tmp_pos] = numbers[mid];
		mid = mid + 1;
		tmp_pos = tmp_pos + 1;
	}
	
	for (i=0; i <= num_elements; i++)
	{
		numbers[right] = temp[right];
		right = right - 1;
	}
}

/********************************************************************************************************************

PERMUTATION ALGORITHMS - these ought to go in benlib sometime

********************************************************************************************************************/

/*
	function to permute an array of scalars v of length n and store
	the result in the array u, according to a 
	permutation vector p with the permutation direction governed by map_flag
 
	map_flag == 0
 u[p[i]] <- v[i]
 
	map_flag != 0
 u[i] <- v[p[i]]
 */
void permute( double *v, double *u, int *p, int n, int map_flag )
{
	int i;
	
	if( !map_flag )
	{
		for( i=0; i<n; i++ )
		{
			u[p[i]] = v[i];
		}
	}
	else
	{
		for( i=0; i<n; i++ )
		{
			u[i] = v[p[i]];
		}
	}
}

/*
	function to permute an array of scalars v of length n and store
	the result back in v, according to a 
	permutation vector p with the permutation direction governed by map_flag
 
	map_flag == 0
 		u[p[i]] <- v[i]
 
	map_flag != 0
 		u[i] <- v[p[i]]
 
	IMPORTANT : This routine trashes to array p, so back it up before
	you start if you want to use it again. This is just really handy for
	doing permutations in situations where moving memory around would be
	a hassle.
 
	This is a bit of a dog of an algorithm, in a best case scenario it is very
	fast, however it has the potential to be very slow.
 */
void permute_inplace( double *v, int *p, int n, int map_flag )
{
	int i, j;
	double tmp;
	
	ASSERT_MSG( map_flag, "permute_inplace() : unable to permute v[i]=u[p[i]] in place, can be done, but needs extra storeage." );
	
	if( map_flag )
	{
		for( i=0; i<n; i++ )
		{
			/**********************************
			u[p[i]] = v[i];
			**********************************/
			
			// note that we don't need to do anything if p[i]==i
			if( p[i]>i )
			{
				tmp = v[i];
				v[i] = v[p[i]];
				v[p[i]] = tmp;
				p[i] = i;
			}
			else if( p[i]<i )
			{
				j = p[i];
				while( p[j]<j )
				{
					j=p[j];
				}
				tmp = v[j];
				v[j] = v[p[j]];
				v[p[j]] = tmp;
				p[j] = j;
			}
		}
	}
}

/*
	function to permute an array of blocks v of length n and store
	the result in the array u, according to a 
	permutation vector p with the permutation direction governed by map_flag
 
	map_flag == 0
 u[p[i]] <- v[i]
 
	map_flag != 0
 u[i] <- v[p[i]]
 */
void permuteB( double *v, double *u, int *p, int n, int map_flag )
{
	int i;
	double *from, *to;
	
	if( !map_flag )
	{
		from = v;
		for( i=0; i<n; i++, from+=(BLOCK_SIZE*BLOCK_SIZE) )
		{
			/**********************************
			u[p[i]] = v[i];
			**********************************/
			to = u + (p[i] BLOCK_M_SHIFT);
			BLOCK_M_COPY( from, to );
		}
	}
	else
	{
		to = u;
		for( i=0; i<n; i++, to+=(BLOCK_SIZE*BLOCK_SIZE) )
		{
			/**********************************
			u[i] = v[p[i]];
			**********************************/
			from = v + (p[i] BLOCK_M_SHIFT);
			BLOCK_M_COPY( from, to );
		}
	}
}

/*
	heapsort routine to sort an array of list of integers
	very fast;
*/
void heapsort_int( int n, int *ra)
{
	int i, ir, j, l, rra;
	
	// adjust for the original fortran node numbering
	ra--;
	
	if( n<2 )
		return;
	
	l=(n >> 1)+1;
	ir=n;
	
	for(;;)
	{
		if( l>1 )
		{
			rra=ra[--l];
		}
		else
		{
			rra=ra[ir];
			ra[ir]=ra[1];
			if( --ir==1 )
			{
				ra[1]=rra;
				break;
			}
		}
		i=l;
		j=l+l;
		while( j<=ir )
		{
			if( j<ir && ra[j]<ra[j+1] )
				j++;
			if( rra<ra[j] )
			{
				ra[i]=ra[j];
				i=j;
				j<<=1;
			}
			else
				break;
		}
		ra[i]=rra;
	}
}

void heapsort_double( int n, double *ra)
{
	int i, ir, j, l;
	double rra;
	
	// adjust for the original fortran node numbering
	ra--;
	
	if( n<2 )
		return;
	
	l=(n >> 1)+1;
	ir=n;
	
	for(;;)
	{
		if( l>1 )
		{
			rra=ra[--l];
		}
		else
		{
			rra=ra[ir];
			ra[ir]=ra[1];
			if( --ir==1 )
			{
				ra[1]=rra;
				break;
			}
		}
		i=l;
		j=l+l;
		while( j<=ir )
		{
			if( j<ir && ra[j]<ra[j+1] )
				j++;
			if( rra<ra[j] )
			{
				ra[i]=ra[j];
				i=j;
				j<<=1;
			}
			else
				break;
		}
		ra[i]=rra;
	}
}

void heapsort_longlong( int n, long long *ra)
{
	int i, ir, j, l;
	long long rra;
	
	// adjust for the original fortran node numbering
	ra--;
	
	if( n<2 )
		return;
	
	l=(n >> 1)+1;
	ir=n;
	
	for(;;)
	{
		if( l>1 )
		{
			rra=ra[--l];
		}
		else
		{
			rra=ra[ir];
			ra[ir]=ra[1];
			if( --ir==1 )
			{
				ra[1]=rra;
				break;
			}
		}
		i=l;
		j=l+l;
		while( j<=ir )
		{
			if( j<ir && ra[j]<ra[j+1] )
				j++;
			if( rra<ra[j] )
			{
				ra[i]=ra[j];
				i=j;
				j<<=1;
			}
			else
				break;
		}
		ra[i]=rra;
	}
}

/*
	sort a list of integers stored in ra. The array index is also sorted in the same order,
	so that index stores the permutation vector used to sort a... if you initialised index
	to be index=[0:n-1]
*/
void heapsort_int_index( int n, int *index, int *ra )
{
	int i, ir, j, l, rri;
	int rra;
	
	index--;
	ra--;
	
	if( n<2 )
		return;
	
	l=(n >> 1)+1;
	ir=n;
	
	for(;;)
	{
		if( l>1 )
		{
			rra=ra[--l];
			rri=index[l]; //
		}
		else
		{
			rra=ra[ir];
			ra[ir]=ra[1];
			rri=index[ir]; //
			index[ir]=index[1]; //
			if( --ir==1 )
			{
				ra[1]=rra;
				index[1]=rri;
				break;
			}
		}
		i=l;
		j=l+l;
		while( j<=ir )
		{
			if( j<ir && ra[j]<ra[j+1] )
				j++;
			if( rra<ra[j] )
			{
				ra[i]=ra[j];
				index[i]=index[j];
				i=j;
				j<<=1;
			}
			else
				break;
		}
		ra[i]=rra;
		index[i]=rri;
	}
}

void heapsort_longlong_index( int n, int *index, long long *ra )
{
	int i, ir, j, l, rri;
	long long rra;
	
	index--;
	ra--;
	
	if( n<2 )
		return;
	
	l=(n >> 1)+1;
	ir=n;
	
	for(;;)
	{
		if( l>1 )
		{
			rra=ra[--l];
			rri=index[l]; //
		}
		else
		{
			rra=ra[ir];
			ra[ir]=ra[1];
			rri=index[ir]; //
			index[ir]=index[1]; //
			if( --ir==1 )
			{
				ra[1]=rra;
				index[1]=rri;
				break;
			}
		}
		i=l;
		j=l+l;
		while( j<=ir )
		{
			if( j<ir && ra[j]<ra[j+1] )
				j++;
			if( rra<ra[j] )
			{
				ra[i]=ra[j];
				index[i]=index[j];
				i=j;
				j<<=1;
			}
			else
				break;
		}
		ra[i]=rra;
		index[i]=rri;
	}
	
}

void heapsort_double_index( int n, int *index, double *ra )
{
	int i, ir, j, l, rri;
	double rra;
	
	index--;
	ra--;
	
	if( n<2 )
		return;
	
	l=(n >> 1)+1;
	ir=n;
	
	for(;;)
	{
		if( l>1 )
		{
			rra=ra[--l];
			rri=index[l]; //
		}
		else
		{
			rra=ra[ir];
			ra[ir]=ra[1];
			rri=index[ir]; //
			index[ir]=index[1]; //
			if( --ir==1 )
			{
				ra[1]=rra;
				index[1]=rri;
				break;
			}
		}
		i=l;
		j=l+l;
		while( j<=ir )
		{
			if( j<ir && fabs(ra[j])<fabs(ra[j+1]) )
				j++;
			if( fabs(rra)<fabs(ra[j]) )
			{
				ra[i]=ra[j];
				index[i]=index[j];
				i=j;
				j<<=1;
			}
			else
				break;
		}
		ra[i]=rra;
		index[i]=rri;
	}
}

void heapsort_int_dindex( int n, double *index, int *ra )
{
	int i, ir, j, l, rra;
	double rri;
	
	index--;
	ra--;
	
	if( n<2 )
		return;
	
	l=(n >> 1)+1;
	ir=n;
	
	for(;;)
	{
		if( l>1 )
		{
			rra=ra[--l];
			rri=index[l]; //
		}
		else
		{
			rra=ra[ir];
			ra[ir]=ra[1];
			rri=index[ir]; //
			index[ir]=index[1]; //
			if( --ir==1 )
			{
				ra[1]=rra;
				index[1]=rri;
				break;
			}
		}
		i=l;
		j=l+l;
		while( j<=ir )
		{
			if( j<ir && ra[j]<ra[j+1] )
				j++;
			if( rra<ra[j] )
			{
				ra[i]=ra[j];
				index[i]=index[j];
				i=j;
				j<<=1;
			}
			else
				break;
		}
		ra[i]=rra;
		index[i]=rri;
	}
}

/*
	takes a vector of BLOCK_SIZE x BLOCK_SIZE block matrices and transposes each one.
	This operation is useful for taking the transpose of a larger block matrix
*/
void block_transpose( double *a, int n )
{
	int i, tmp;
	
	for( i=0; i<n; i++, a+=(BLOCK_SIZE*BLOCK_SIZE)  )
	{
		BLOCK_M_TRANSPOSE( a, tmp );
	}
}


/***********************************************
	Routines for manipulating nodes
***********************************************/
Tnode_ptr node_new( int n, int tag, int block, Tnode_ptr prev, Tnode_ptr next )
{
	Tnode_ptr d;
	
	d = (Tnode_ptr)malloc( sizeof(Tnode) );
	d->next = next;
	d->prev = prev;
	if( next!=NULL )
		next->prev = d;
	if( prev!=NULL )
		prev->next = d;
	d->indx = (int *)malloc( sizeof(int)*n );
	if( !block )
		d->dat = (double *)malloc( sizeof(double)*n );
	else
		d->dat = (double *)malloc( sizeof(double)*n BLOCK_M_SHIFT );
	d->n = n;
	d->tag = tag;
	return d;
}

void node_list_init( Tnode_list_ptr L, int block )
{	
	L->n = 0;
	L->block = block;
	L->start = NULL;
	L->head = NULL;
	L->opt = NULL;
}

/*
insert a new node to a list.
*/
int node_list_add( Tnode_list_ptr L, int n, int tag )
{
	// locate where to add the node in the list
	Tnode_ptr node;
	
	node = L->opt;
	L->n++;
	
	// the list is empty, so add the node as is
	if( node==NULL )
	{
		L->head = node_new( n, tag, L->block, NULL, NULL );
		L->opt = L->head;
		L->start = L->head;
		return 1;
	}
	
	// the new node goes at the start;
	if( tag < L->start->tag )
	{
		L->start = node_new( n, tag, L->block, NULL, L->start );
		L->opt = L->start;
		return 1;
	}
	
	// the new node goes at the end
	if( tag > L->head->tag )
	{
		L->head = node_new( n, tag, L->block, L->head, NULL );
		L->opt = L->head;
		return 1;
	}
	
	// the new node is below the current optimal insertion range
	if( tag < node->tag )
	{
		while( tag < node->tag  )
		{
			node = node->prev;
		}
		L->opt = node_new(  n, tag, L->block, node, node->next );
		return 1;
	}
	
	// the new node is above the current optimal insertion range
	if( tag > node->tag )
	{
		while( tag > node->tag  )
		{
			node = node->next;
		}
		L->opt = node_new(  n, tag, L->block, node->prev, node );
		return 1;
	}
	
	ERROR( "node_list_add() : attempt to add node with non-unique tag" );
	return 0;
}

void node_list_free( Tnode_list_ptr L )
{
	Tnode_ptr next;
	Tnode_ptr node;
	
	node = L->start;
	while( node!=NULL )
	{
		free( node->dat );
		free( node->indx );
		next = node->next;
		free( node );
		node = next;
	}
	
	L->n = 0;
	L->opt =  NULL;
	L->start = NULL;
	L->head = NULL;
}

/*
	pop node from the head of L 
*/
Tnode_ptr node_list_pop( Tnode_list_ptr L )
{
	Tnode_ptr node;
	
	node = L->head;
	if( node==NULL )
	{
		return node;
	}
	
	L->n--;
	
	if( !L->n )
	{
		L->start = L->head = NULL;
		L->opt = L->head;
	}
	else
	{
		L->head = node->prev;
		L->head->next = NULL;
		if( L->opt==node )
			L->opt = L->head;
	}
	
	return node;
}

/*
	push node to the head of L
*/
void node_list_push_head( Tnode_list_ptr L, Tnode_ptr node )
{
	if( L->head==NULL )
	{
		L->head = L->start = L->opt = node;
		node->next = node->prev = NULL;
		L->n++;
		return;
	}
	L->n++;
	L->opt = node;
	node->prev = L->head;
	node->next = NULL;
	L->head->next = node;
	L->head = node;
}

/*
	push node to the start of L
*/
extern void node_list_push_start( Tnode_list_ptr L, Tnode_ptr node )
{
	if( L->start==NULL)
	{
		L->start = L->head = L->opt = node;
		L->n++;
		node->next = node->prev = NULL;
		return;
	}
	L->n++;
	L->opt = node;
	node->next = L->start;
	L->start->prev = node;
	node->prev = NULL;
	L->start = node;
}

/*
	a routine to verify that a node list is valid, that is all nodes are sorted in ascending order,
	and all pointers work etc.
*/
int node_list_verify( FILE *stream, Tnode_list_ptr L )
{
	Tnode_ptr node;
	int count, isopt=0, inorder=1, i;
	
	// print out an initial header
	if( stream!=NULL )
	{
		fprintf( stream, "\n================================================\n" );
		fprintf( stream, "Starting node list verification : list of %d entries\n\n", L->n );
		//fprintf( stream, "\t->start = %8X\n\t->head  = %8X\n\t->opt   = %8X\n\n", (unsigned int)L->start , (unsigned int)L->head , (unsigned int)L->opt );
		fflush( stream );
	}
		

	
	// test for an empty list
	if( !L->n )
	{
		if( stream!=NULL )
		{
			fprintf( stream, "\tEmpty list\n" );
		}
			
		if( L->opt!=NULL || L->head!=NULL || L->start!=NULL )
		{
			if( stream!=NULL )
				fprintf( stream, "\tERROR : empty list has non NULL pointers\n" );
			return 0;
		}
		if( stream!=NULL )
			fprintf( stream, "================================================\n" );
		return 1;
	}
	
	// test out the list node ordering and pointers
	if( stream!=NULL )
		fprintf( stream, "Checking node order and pointers\n" );
	
	// test a non empty list
	node = L->start;
	count = 0;
	if( stream!=NULL )
		fprintf( stream, "\tNon-empty list\n" );
	while( node!=NULL && count<L->n )
	{			
		if( node==L->opt )
			isopt=1;
		
		if( stream!=NULL );
			//fprintf( stream, "\t\tnode %d with tag %d\t<%8X> <%8X> <%8X>\n", count, node->tag, (unsigned int)node->prev, (unsigned int)node, (unsigned int)node->next );
		
		if( !count )
		{
			if( node->prev!=NULL )
			{
				if( stream!=NULL )
					fprintf( stream, "\tERROR : start node has non-NULL ->prev pointer\n" );
				return 0;
			}
		}
		if( count==L->n-1 )
		{
			if( node->next!=NULL)
			{
				if( stream!=NULL )
					fprintf( stream, "\tERROR : head node has non-NULL ->next pointer\n" );
				return 0;
			}
		}
		else
		{
			if( node->next==NULL)
			{
				if( stream!=NULL )
					fprintf( stream, "\tERROR : This node points to a NULL next node, but should not be the last\n" );
				return 0;
			}
			if( node->next->prev != node )
			{
				if( stream!=NULL )
				{
					fprintf( stream, "\tERROR : This node does not doubly-link with the next properly\n" );
					//fprintf( stream, "\t\tnode %d with tag %d\t<%8X> <%8X> <%8X>\n", count, node->next->tag, (unsigned int)node->next->prev, (unsigned int)node->next, (unsigned int)node->next->next );
				}
				
				return 0;
			}
			if( !(node->tag<node->next->tag) )
				inorder = 0;
		}
		
		count++;
		node = node->next;
	}
	
	if( !inorder )
	{
		if( stream!=NULL )
			fprintf( stream, "\tERROR : tags are not in order\n" );
		return 0;
	}
	if( count==L->n && node!=NULL )
	{
		if( stream!=NULL )
			fprintf( stream, "\tERROR : L->n and the number of nodes does not match up : more nodes than L->n\n" );
		return 0;
	}
	if( count<L->n-1 )
	{
		if( stream!=NULL )
			fprintf( stream, "\tERROR : L->n and the number of nodes does not match up : less nodes than L->n\n" );
		return 0;
	}
	
	if( !isopt )
	{
		if( stream!=NULL )
			fprintf( stream, "\tERROR : L->opt is not in the list" );
		return 0;
	}
	
	// touch all of the data
	if( stream!=NULL )
		fprintf( stream, "Touching list data\n" );
	node = L->start;
	count = 0;
	
	while( node!=NULL)
	{
		int itmp;
		double dtmp;
		
		if( stream!=NULL )
			fprintf( stream, "\tnode %d with %d entries\n", count, node->n );
		if( node->indx==NULL || node->dat==NULL )
		{
			if( stream!=NULL )
				fprintf( stream, "\tERROR : null data/index pointers\n" );
			return 0;
		}
		for( i=0; i<node->n; i++ )
		{
			itmp = node->indx[i];
			dtmp = node->dat[i];
			node->indx[i] = itmp;
			node->dat[i] = dtmp;	
		}
		count++;
		node = node->next;
	}
	
	if( stream!=NULL )
	{
		fprintf( stream, "\nFinished node list verification\n" );
		fprintf( stream, "================================================\n" );
	}
	return 1;
}

void print_nodes( FILE *stream, char *msg, Tnode_list_ptr L )
{
	Tnode_ptr node;
	int i;
	
	node = L->start;
	while( node!=NULL )
	{
		fprintf( stream, "%s : %d\t-", msg, node->tag );
		for( i=0; i<node->n; i++ )
			fprintf( stream, "\t%d", node->indx[i] );
		fprintf( stream, "\n" );
		node = node->next;
	}
}

/**************************************************************************************
 *		returns the system time for the process which we are currently running 
 *		time is in seconds, as a double 
 **************************************************************************************/
double get_time( void )
{
	struct rusage ru;
	
	getrusage( RUSAGE_SELF, &ru );
	
	return( (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec/1000000. );
}


/*************************************************************************
* This function does a binary search on an array for a key and returns
* the index
*
* returns -1 if key is not found in the list
**************************************************************************/
int binary_search(int n, int *array, int key)
{
	int a=0, b=n, c;
	
	while (b-a > 8) {
		c = (a+b)>>1;
		if (array[c] < key)
			b = c;
		else
			a = c;
	}
	
	for (c=a; c<b; c++) {
		if (array[c] == key)
			return c;
	}
		
	return -1;
}

/*************************************************************************
* a specialist routine used in vec_drop_block()
*
* used for searching an array of doubles. The doubles in array are assumed
* to have been sorted in descending order. The algorithm does not search for
* an exact match for key, instead it finds the entries in array that bracket
* key. the returned value is the index of the largest entry in array that is
* smaller than or equal to key.
**************************************************************************/
int binary_search_double_bracket(int n, double *array, double key)
{
	int a=0, b=n-1, c;
	
	if( !n )
		return -1;
	if( array[0]<key )
		return -1;
	if( array[n-1]>=key )
		return n-1;
	
	while (b-a > 4) 
	{
		c = (a+b)>>1;
		if (array[c] < key)
			b = c;
		else
			a = c;
	}
	for (c=a; c<b; c++) 
	{
		if (array[c] >= key && array[c+1] < key )
			return c;
	}
	
	return c;
}

