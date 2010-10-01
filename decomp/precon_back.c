/*
 *	precon.c
 */

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "ben_mpi.h"
#include "benlib.h"
#include "linalg_mpi.h"
#include "linalg_dense_mpi.h"
#include "parmetis.h"
#include "linalg.h"
#include "indices.h"
#include "fileio.h"
#include "MR.h"
#include "precon.h"
#include "gmres.h"
#include "ILU.h"


#define OUTPUTNODEORDER  \
for( k=0; k<n_dom; k++ )\
{\
	if( k==this_dom )\
	{\
		fprintf( fid, "Process %d : Finished ParMetis\n", this_dom );\
			for( i=0; i<A->mtx.nrows; i++ )\
			{\
				fprintf( fid, "\t[%d, %d]\n", P->forward.part[i], P->forward.indx[i] );\
			}\
			for( i=0; i<n_dom; i++ )\
			{\
				if( P->forward.starts[i]!=P->forward.starts[i+1] )\
					fprintf( fid, "\tI own %d nodes belonging to dom %d indexed (%d, %d)\n", P->forward.counts[i], i, P->forward.starts[i], P->forward.starts[i+1]-1 );\
						else\
							fprintf( fid, "\tI own %d nodes belonging to dom %d\n", P->forward.counts[i], i );\
			}\
			fprintf( fid, "\n" );\
				fflush( fid );\
	}\
	MPI_Barrier( This->comm );\
}\
fprintf( fid, "\n========================================\nDISTRIBUTING NODES\n========================================\n" );

#define PRINTGLOBALCONNECTIONS \
fprintf( fid, "My copy of the global connection map is somewhat like :\n\n" );\
for( i=0; i<n_dom; i++ )\
{\
	for( j=0; j<n_dom; j++ )\
		fprintf( fid, "\t%d", P->domains.map[i+j*n_dom] );\
			fprintf( fid, "\n" );\
}

// prototype for routine only used herein
int tag_create(  int source, int node, int msgtype );
// sets up gmres parameters for schur lowest level solver
void GMRES_setup_schur( Tgmres_ptr g );
// print out the indices of nz entries in integer array v of length n
void find_i( int *v, int n );
void find_v( int *v, double *u, int n );
void print_sparsity( double *v, int n, char *tag );
void print_sparsity_nz( double *v, int n, char *tag );


void precon_print_name( int type )
{
	switch( type )
	{
		case PRECON_JACOBI :
		{
			printf( "PRECON_JACOBI\n" ); break;
		}
		case PRECON_SCHUR_SPLIT  :
		{
			printf( "PRECON_SCHUR_SPLIT\n" ); break;
		}
		case PRECON_NONE  :
		{
			printf( "PRECON_NONE\n" ); break;
		}
		case PRECON_ILUT :
		{
			printf( "PRECON_ILUT\n" ); break;
		}
		case PRECON_ILU0 :
		{
			printf( "PRECON_ILU0\n" ); break;
		}
		default :
		{
			printf( "PRECON_UNKNOWN\n" ); break;
		}
	}
	
}

/******************************************************************************************
*	precon_init()
*
*   initialise a preconditioner type
*
******************************************************************************************/
void precon_init( Tprecon_ptr precon, int type, int parallel, TMPI_dat_ptr This )
{
	switch( type )
	{
		case PRECON_JACOBI :
		{
			precon->type = type;
			precon->parallel = parallel;
			precon->preconditioner = malloc( sizeof(Tvec_dist) );
			BMPI_copy( This, &precon->This );
			break;
		}
		case PRECON_SCHUR_SPLIT  :
		{
			precon->type = type;
			precon->parallel = parallel;
			precon->preconditioner = malloc( sizeof(Tprecon_Schur) );
			BMPI_copy( This, &precon->This );
			break;
		}
		case PRECON_NONE  :
		{
			precon->type = type;
			precon->parallel = parallel;
			precon->preconditioner = NULL;
			BMPI_copy( This, &precon->This );
			break;
		}
		case PRECON_ILUT :
		{
			precon->type = type;
			precon->parallel = parallel;
			precon->preconditioner = malloc( sizeof(Tprecon_ILU) );
			BMPI_copy( This, &precon->This );
			break;
		}
		case PRECON_ILU0 :
		{
			precon->type = type;
			precon->parallel = parallel;
			precon->preconditioner = malloc( sizeof(Tprecon_ILU0) );
			BMPI_copy( This, &precon->This );
			break;
		}
		default :
		{
			fprintf( stderr, "ERROR : halting, invalid preconditioner type in precon_init()\n\n" );
			MPI_Abort( This->comm, 1 );
		}
	}
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
			precon_jacobi_apply( (Tvec_dist_ptr)P->preconditioner, x, y );
			break;
		}
		case PRECON_SCHUR_SPLIT :
		{
			precon_schur_apply( (Tprecon_Schur_ptr)P->preconditioner, x, y );
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

/******************************************************************************************
*	precon_Schur_init()
*
*	perform basic initialisation of a Schur  compliment preconditioner.
*	The preconditioner is for a matrix distributed over This and is for
*	the matrix A.
******************************************************************************************/
void precon_Schur_init( Tprecon_Schur_ptr P, Tmtx_CRS_dist_ptr A  )
{
	// check that A has been initialised
	ASSERT_MSG( A->init, "precon_Schur_init() : trying to initialise preconditioner for an uninitialised matrix" );
	
	if( P->init )
		precon_schur_free( P );
	
	// initialise the data structure
	P->init    = 1;
	P->n_in    = 0;
	P->n_bnd   = 0;
	P->n_local = 0;
	P->n_neigh = 0;
	P->A.init  = 0;
	P->A.init  = 0;
	P->S.init  = 0;
	P->part_g  = (int*)malloc( sizeof(int)*A->nrows );
	P->q       = (int*)malloc( sizeof(int)*A->nrows );
	P->p	   = NULL;
	BMPI_copy( &A->This, &P->This );
}


/******************************************************************************************
*		mtx_CRS_dist_split()
*
*		take a distributed CRS matrix A and perform domain decomp
*		and split it into the Schur form in the precondioner P
*******************************************************************************************/
void mtx_CRS_dist_split( Tmtx_CRS_dist_ptr A, Tprecon_Schur_ptr P )
{
	FILE *fid;
	char fname[50];
	int i, j, k, pos, this_dom, dom, n_dom, Fnnz, Bnnz, Cnnz, Ennz, tdom, sdom, r_pos, n_nodes;
	int tag, index_pos, recv_nodes, is_boundary, nnz, success, success_global;
	int n_in, n_bnd, n_node, row_nnz, nodes_max, nnz_max;
	int *is_neigh, *back_counts, *Eijnnz, *index_pack, *tag_pack, *nnz_pack;
	int *p, *int_temp, *in_split, **bnd_split;
	int block, n_neigh, connect_dom, max_node_nnz=0, spot, this_dom_pos, dom_start;
	double *dbl_temp, *nz_pack;
	Tnode_ptr  node=NULL;
	Tnode_list in, bnd, nodes;
	TMPI_dat_ptr This;
	MPI_Request *request;
	MPI_Status *status;
	
	/*
	 *		Setup the diagnostic file for This thread
	 *		
	 *		This file contains diagnostic output describing the communication
	 *		between processes/domains and the domain decompt method
	 */
	
	This = &A->This;
	sprintf( fname, "P%d_split.txt", This->this_proc );
	fid = fopen( fname, "w" );
	
	/*
	 *		Setup initial data stuff
	 */
	
	// setup variables
	this_dom    = This->this_proc;
	n_dom       = This->n_proc;
	block       = A->mtx.block;
	back_counts = (int*)malloc( sizeof(int)*n_dom );
	
	// initialise data types
	node_list_init( &in,    block );
	node_list_init( &bnd,   block );
	node_list_init( &nodes, block );
	
	// domain decomp using ParMETIS
	mtx_CRS_dist_domdec( A, &P->forward, P->part_g );
	// ouput the permution to disk
	if( !this_dom )
	{
		FILE *ffid;
		
		ffid = fopen( "p.txt", "w" );
		for( i=0; i<A->nrows; i++ )
		{
			fprintf( ffid, "%d\n", P->part_g[i] );
		}
		fclose( ffid );
	}

	// output the new node order
	OUTPUTNODEORDER;
	
	
	/* 
	 *		Send and recieve nodes
	 *
	 *		The processors exchange nodes with one another so that at completion of This
	 *		loop each processor has only the nodes corresponding to its domain.
	 *		On each iteration of the loop P(i) will give nodes to P(i+k) and recieve
	 *		nodes from P(i-k), allowing all processes to continue communicating 
	 *		constantly.
	 *
	 *		The nodes are stored in node linked lists, not in CRS format, This allows
	 *		easy sorting and manipulation of the nodes.
	 */
	
	request = malloc( 20*sizeof(MPI_Request) );
	status  = malloc( 20*sizeof(MPI_Status) );
	for( i=0; i<20; i++ )
		status[i].MPI_ERROR = MPI_SUCCESS;
	nodes_max = 0;
	for( k=0; k<n_dom; k++ )
		if( nodes_max<P->forward.counts[k] )
			nodes_max = P->forward.counts[k];
	tag_pack   = (int*)malloc( nodes_max*sizeof(int) );
	nnz_pack   = (int*)malloc( nodes_max*sizeof(int) );
	
	nnz_max    = A->mtx.nnz;
	nz_pack    = (double*)malloc( nnz_max*sizeof(double) );
	index_pack = (int*)malloc( nnz_max*sizeof(int) );
	
	for( k=0; k<n_dom; k++ )
	{
		r_pos = 0;
		
		// determine the target and source domains
		tdom = k +  this_dom;
		if( tdom>=n_dom )	
			tdom -= n_dom;
		
		sdom = this_dom-k;
		if( sdom<0 )		
			sdom += n_dom;
		
		/*
		 *		Pack the data for sending
		 */
		
		// the number of nodes to send to sdom
		n_nodes = P->forward.counts[tdom];	
		nnz = 0;
		
		
		for( pos=0, i=P->forward.starts[tdom]; i<P->forward.starts[tdom+1]; pos++, i++ )
		{
			index_pos = A->mtx.rindx[P->forward.indx[i]];
			
			tag_pack[pos]   = A->vtxdist[this_dom] + P->forward.indx[i];
			nnz_pack[pos]   = A->mtx.rindx[P->forward.indx[i]+1] - index_pos;
			memcpy( index_pack + nnz, A->mtx.cindx + index_pos, nnz_pack[pos]*sizeof(int) );
			memcpy( nz_pack + nnz,    A->mtx.nz + index_pos,    nnz_pack[pos]*sizeof(double) );
						
			nnz+=nnz_pack[pos];
		}
		
		fprintf( fid, "P%d : sending to P%d : %d nodes with %d indices\n", this_dom, tdom, n_nodes, nnz );		
		/*
		 *		Send information to target Pid
		 */
		
		// send the number of nodes being passed
		MPI_Isend( &n_nodes, 1,      MPI_INT, tdom, TAG_NODES, This->comm, request+r_pos );
		r_pos++;
		
		if( n_nodes )
		{
			// tags
			MPI_Isend( tag_pack, n_nodes, MPI_INT, tdom, TAG_TAG, This->comm, request+r_pos );
			r_pos++;
			
			// nnz
			MPI_Isend( nnz_pack,   n_nodes, MPI_INT, tdom, TAG_NNZ, This->comm, request+r_pos );
			r_pos++;
			
			// data
			MPI_Isend( index_pack,  nnz,     MPI_INT, tdom, TAG_INDEX, This->comm, request+r_pos );
			r_pos++;
			
			// data
			MPI_Isend( nz_pack,     nnz,     MPI_DOUBLE, tdom, TAG_NZ, This->comm, request+r_pos );
			r_pos++;
		}
			
		
		/*
		 *		Receive data from the source Pid
		 */
		
		//  wait for the source to have sent its data to you, if you jump the gun here
		//	things go pear shaped pretty quickly
		MPI_Barrier( This->comm );
		
		// number of nodes
		MPI_Irecv( &recv_nodes, 1, MPI_INT, sdom, TAG_NODES , This->comm, request+r_pos );
		r_pos++;
		
		// check that we aren't going to run out of memory
		if( recv_nodes>nodes_max )
		{
			nodes_max = recv_nodes;
			tag_pack = (int*)realloc( tag_pack, nodes_max*sizeof(int) );
			nnz_pack = (int*)realloc( nnz_pack, nodes_max*sizeof(int) );
		}
		
		back_counts[ sdom ] = recv_nodes;
		
		if( recv_nodes )
		{
			// tags
			MPI_Irecv( tag_pack, recv_nodes, MPI_INT, sdom, TAG_TAG, This->comm, request+r_pos );
			r_pos++;
			
			// nnz
			MPI_Irecv( nnz_pack, recv_nodes, MPI_INT, sdom, TAG_NNZ, This->comm, request+r_pos );
			r_pos++;
			
			// calculate the number of indices being received
			nnz = 0;
			for( i=0; i<recv_nodes; i++ )
				nnz += nnz_pack[i];
			//printf( "P%d : receiving %d indices from P%d\n", this_dom, nnz, sdom );
			
			// check that we have enough memory in the buffer
			if( nnz>nnz_max )
			{
				nnz_max = nnz;
				index_pack =    (int*)realloc( index_pack, nnz_max*sizeof(int) );
				nz_pack    = (double*)realloc( nz_pack,    nnz_max*sizeof(double) );
			}
			
			// indices
			MPI_Irecv( index_pack,  nnz, MPI_INT, sdom, TAG_INDEX, This->comm, request+r_pos );
			r_pos++;
			
			// nz values
			MPI_Irecv( nz_pack,  nnz,   MPI_DOUBLE, sdom, TAG_NZ, This->comm, request+r_pos );
			r_pos++;
		}
		
		// wait for all communication to finish
		for( i=0; i<r_pos; i++ )
		{
			MPI_Wait( request+i, status+i );
			if( status[i].MPI_ERROR!=MPI_SUCCESS)
				printf( "P%d -- WARNING : status for a message %d was not MPI_SUCCESS\n", this_dom, i );
		}
				
		/*
		 *		Unpack data into node list
		 */
		for( i=0, nnz=0; i<recv_nodes; i++ )
		{
			
			// allocate memory for the new node
			if( !node_list_add( &nodes, nnz_pack[i], tag_pack[i] ) )
				fprintf( stderr, "\tP%d : ERROR, unable to add node with tag %d\n", this_dom, tag );
			
			// copy over the indices and nz values
			memcpy( nodes.opt->indx, index_pack + nnz, sizeof(int)*nnz_pack[i] );
			memcpy( nodes.opt->dat, nz_pack + nnz, sizeof(double)*nnz_pack[i] );
			
			nnz += nnz_pack[i];
		}
		
		fprintf( fid, "\n" );
	}
	node = nodes.start;
	while( node!=NULL )
	{
		if( node->n>max_node_nnz )
			max_node_nnz = node->n;
		node = node->next;
	}
	free( index_pack );
	free( nnz_pack );
	free( tag_pack );
	free( nz_pack );
	free( request );
	free( status  );
	
	fprintf( fid, "\n========================================\nFINISHED DISTRIBUTING NODES\n========================================\n" );
	
	// everyone checks that they have recieved valid node lists
	success = node_list_verify( NULL, &nodes );
	MPI_Allreduce( &success, &success_global, 1, MPI_INT, MPI_LAND, This->comm );
	if( !success_global )
	{
		fprintf( fid, "ERROR : one of the processes had an invalid node list\n" );
		return;
	}
	
	/*
	 *		Determine the internal and boundary nodes for This domain.
	 *
	 *		Internal and boundary nodes are sorted into the lists in and out
	 *		respectively. The element is_neigh[i] records how many edges This
	 *		domain shairs with domain i. Remember, the number of edges is NOT
	 *		the number of nodes connected to. Think of is_neigh[i] as the nnz
	 *		in the Ei for This domain.
	 */
	is_neigh = (int *)calloc( n_dom, sizeof(int) );
	nnz = 0;
	p = P->part_g;
	pos = 1;
	while( (node = node_list_pop( &nodes )) != NULL )
	{
		is_boundary = 0;
		
		// does the node contain references to external nodes?
		for( i=0; i<node->n; i++ )
		{
			// is This an external reference?
			connect_dom = p[node->indx[i]];
			if( connect_dom!=this_dom )
			{
				// it is so add the domain to our list of neighbours
				is_neigh[connect_dom]++;
				
				// flag This node as being on the boundary
				is_boundary = 1;
			}
		}
		
		// store the node in the relevant list
		if( is_boundary )
		{
			node_list_push_start( &bnd, node );
		}
		else
		{
			node_list_push_start( &in, node );
		}
		nnz += node->n;
		pos ++;
		fprintf( fid, "\n" );
	}
	
	// determine the number of nodes that are in/out
	P->n_in  = n_in   = in.n;
	P->n_bnd = n_bnd  = bnd.n;
	P->n_local = n_in + n_bnd;
	n_node  = n_in + n_bnd;
	is_neigh[this_dom] = n_in;
	
	// determine how many domains we are connected to
	n_neigh = 0;
	for( k=0; k<n_dom; k++ )
	{
		if( is_neigh[this_dom] )
			n_neigh++;
	}
	P->n_neigh = n_neigh;
	
	// print out the interior and boundary node data
	fprintf( fid, "\nFinished categorising nodes : I have %d interior and %d boundary nodes\n", in.n, bnd.n );
	print_nodes( fid, "interior", &in );
	print_nodes( fid, "boundary", &bnd );
	
	/*
	 *		Processes send one-another information on who they are connected to
	 *		
	 *		This information is stored in map, where map[i*ndom:(i+1)*ndom-1] contains the is_neigh
	 *		array for domain i.
	 *
	 *		The processors let everyone else know how many nodes they now have, allowing each
	 *		process to build a new vtxdist for the new domain decomposition.
	 */
	domain_init( &P->domains, This, n_node );
	
	// MPI command to distribute information
	MPI_Allgather( is_neigh, n_dom, MPI_INT, P->domains.map,     n_dom, MPI_INT, This->comm );
	
	// output the map to the output file
	// PRINTGLOBALCONNECTIONS;
	
	// determine vtxdist for This distribution
	int_temp = (int *)malloc( sizeof(int)*n_dom );
	MPI_Allgather( &n_node,  1,     MPI_INT, int_temp, 1,     MPI_INT, This->comm );
	
	P->domains.vtxdist[0] = 0;
	for( i=1; i<=n_dom; i++ )
	{
		P->domains.vtxdist[i] = P->domains.vtxdist[i-1]+int_temp[i-1];
	}
	free( int_temp );
	
	// update all processors on the numbner of interior and boundary nodes
	// in each domain
	MPI_Allgather( &n_in,  1, MPI_INT, P->domains.n_in,  1, MPI_INT, This->comm );
	MPI_Allgather( &n_bnd, 1, MPI_INT, P->domains.n_bnd, 1, MPI_INT, This->comm );
	
	/*
	 *  Find the new node order of This matrix, and store in the backwards distribution structure
	 *
	 *  among other things This information is used when passing vectors between processors on the bsub
	 *  phase of preconditioning
	 */
	
	// intialise the data structure
	distribution_init( &P->backward, n_dom, n_node, n_neigh );
	for( pos=0, k=0; pos<n_neigh; k++ )
	{
		if( is_neigh[k] )
			P->backward.neighbours[pos++] = k;
	}
	
	// copy the tags and other data into the distribution arrays
	for( i=0; i<n_node; i++ )
		P->backward.indx[i] = i;
	node = in.start;
	for( i=0; i<n_in; i++ )
	{
		P->backward.part[i] = P->backward.ppart[i] = node->tag;
		node = node->next;
	}
	node = bnd.start;
	for( ; i<n_node; i++ )
	{
		P->backward.part[i] = P->backward.ppart[i] = node->tag;
		node = node->next;
	}
	
	// sort the part so that indx holds the permution vector
	heapsort_int_index( n_node, P->backward.indx, P->backward.part);
	
	// how many nodes were received from each domain has already been calculated
	// so just swap it into the distribution array
	free( P->backward.counts );
	P->backward.counts    = back_counts;
	P->backward.starts[0] = 0;
	for( i=1; i<=n_dom; i++ )
	{
		P->backward.starts[i] = P->backward.starts[i-1] + P->backward.counts[i-1];
	}
	
	/* 
	 *		calculate and store the global permutation vector
	 */
	
	// allocate memory for temporary working arrays
	int_temp = (int *)malloc( sizeof(int)*n_dom );
	p        = (int *)malloc( sizeof(int)*A->nrows );
	
	// determine how many nodes are stored on each processor
	for( i=0; i<n_dom; i++ )
		int_temp[i] = P->domains.vtxdist[i+1] - P->domains.vtxdist[i];
	
	// gather each domain's local ordering for the original node tags
 	MPI_Allgatherv( P->backward.ppart, int_temp[this_dom], MPI_INT, p, int_temp, P->domains.vtxdist, MPI_INT, This->comm );
	
	// convert into global permutation
	for( i=0; i<A->nrows; i++ )
		P->q[p[i]] = i;
		
	// free temporary work arrays
	free( int_temp );
	free( p );
	
	/*
	 *		Store the split of A
	 *
	 *		Done in two steps :
	 *			1.  all indices in nodes are changed to new order and sorted.
	 *				during This the number of nonzero entries in each of the split
	 *				matrices is calculated.
	 *			2.  memory is allocated for the split matrices and the node data is copied
	 *				into the splits.
	 *
	 */
	
	/* 
	 *		do the interior nodes first, these contribute to the matrices B and E
	 */
	
	// setup for the loop
	block = in.block;
	k = P->domains.vtxdist[this_dom];
	j = 0;
	Ennz = Bnnz = 0;
	int_temp = (int *)malloc( sizeof(int)*max_node_nnz );
	in_split = (int *)malloc( sizeof(int)*n_in );
	fflush( fid );
	
	// loop through the nodes
	node = in.start;
	while( node!=NULL )
	{
		// change the node indices
		for( i=0; i<node->n; i++ )
			node->indx[i] = P->q[node->indx[i]] - k; 
		
		// sort the indices, keeping a permutation array
		for( i=0; i<node->n; i++ )
			int_temp[i] = i;
		heapsort_int_index( node->n, int_temp, node->indx);
		
		// sort the nonzero values according to the permuation array
		if( block )
		{
			dbl_temp = (double *)malloc( sizeof(double)*(node->n BLOCK_M_SHIFT) );
			permuteB( node->dat, dbl_temp, int_temp, node->n, 1 );
		}
		else
		{
			dbl_temp = (double *)malloc( sizeof(double)*node->n );
			permute( node->dat, dbl_temp, int_temp, node->n, 1 );
		}
		free( node->dat );
		node->dat = dbl_temp;
		
		// update the split nnz values
		pos=0;
		while( pos<node->n && node->indx[pos]<n_in )
			pos++;
		Bnnz += pos;
		Ennz += (node->n - pos);
		in_split[j] = pos;
		
		// select the next node
		node = node->next;
		j++;
	}
	
	/*
	 *  now make the matrices B and E
	 */
	
	// allocate  memory
	mtx_CRS_init( &P->A.B, n_in, n_in,  Bnnz, block );
	mtx_CRS_init( &P->A.E, n_in, n_bnd, Ennz, block );
	
	// add rows one at a time
	P->A.B.rindx[0] = 0;
	P->A.E.rindx[0] = 0;
	node = in.start;
	j = 0;
	while( node!=NULL )
	{
		// add row to B
		k = in_split[j];
		P->A.B.rindx[j+1] = P->A.B.rindx[j] + k;
		if( k )
		{
			memcpy( P->A.B.cindx + P->A.B.rindx[j], node->indx, k*sizeof(int) );
			if( block )
				memcpy( P->A.B.nz + (P->A.B.rindx[j] BLOCK_M_SHIFT), node->dat, (k BLOCK_M_SHIFT)*sizeof(double)  );
			else
				memcpy( P->A.B.nz + P->A.B.rindx[j], node->dat, k*sizeof(double) );
		}
		
		// add row to E
		k = (node->n - in_split[j]);
		P->A.E.rindx[j+1] = P->A.E.rindx[j] + k;
		if( k )
		{
			// make cindx values for E start at zero
			for( i=in_split[j]; i<node->n; i++ )
				node->indx[i] -= n_in;
			memcpy( P->A.E.cindx + P->A.E.rindx[j], node->indx + in_split[j], k*sizeof(int) );
			if( block )
				memcpy( P->A.E.nz + (P->A.E.rindx[j] BLOCK_M_SHIFT), node->dat + (in_split[j] BLOCK_M_SHIFT), (k BLOCK_M_SHIFT)*sizeof(double) );
			else
				memcpy( P->A.E.nz + P->A.E.rindx[j], node->dat + in_split[j], k*sizeof(double)  );
		}
		
		// select the next node
		node = node->next;
		j++;
	}
	
	// free up working arrays
	free( in_split );
	
	// free up memory for interior node linked list
	node_list_free( &in );
	
	/* 
	 *		do the exterior nodes, these contribute to the matrices F, C and the Eij
	 */
	
	// allocate memory for temporary arrays
	in_split  = (int *)malloc( sizeof(int)*n_bnd );
	Eijnnz    = (int *)calloc( n_neigh, sizeof(int) );
	bnd_split = (int **)malloc( sizeof(int*)*n_bnd );
	for( k=0; k<n_bnd; k++ )
		bnd_split[k] = (int *)malloc( sizeof(int)*(n_neigh+1) );
	
	// intialise loop variables
	j = 0;
	dom_start = P->domains.vtxdist[this_dom];
	this_dom_pos = 0;
	Fnnz = Cnnz = 0;
	while( P->backward.neighbours[this_dom_pos] != this_dom )
		this_dom_pos++;
	
	// loop over nodes
	node = bnd.start;
	while( node!=NULL )
	{		
		// change the node indices
		for( i=0; i<node->n; i++ )
			node->indx[i] = P->q[node->indx[i]]; 
				
		// sort the indices, keeping a permutation array
		for( i=0; i<node->n; i++ )
			int_temp[i] = i;
		heapsort_int_index( node->n, int_temp, node->indx);
		
		// sort the nonzero values according to the permuation array
		if( block )
		{
			dbl_temp = (double *)malloc( sizeof(double)*(node->n BLOCK_M_SHIFT) );
			permuteB( node->dat, dbl_temp, int_temp, node->n, 1 );
		}
		else
		{
			dbl_temp = (double *)malloc( sizeof(double)*node->n );
			permute( node->dat, dbl_temp, int_temp, node->n, 1 );
		}
		free( node->dat );
		node->dat = dbl_temp;

		// determine the which elements belong to which split
		pos = 0;
		for( i=0; i<n_neigh; i++ )
		{
			bnd_split[j][i]=pos;
			dom = P->backward.neighbours[i];
			spot = P->domains.vtxdist[dom+1];
			while( pos<node->n && node->indx[pos] < spot )
				pos++;
		}
		bnd_split[j][n_neigh] = pos;
		
		// work specifically on the diagonal split
		pos = bnd_split[j][this_dom_pos];
		while( pos<n_node && node->indx[pos]-dom_start < n_in )
			pos++;
		in_split[j] = pos;
		
		// update the nnz counts
		for( i=0; i<n_neigh; i++ )
			Eijnnz[i] += bnd_split[j][i+1]-bnd_split[j][i];
		
		Fnnz += in_split[j] - bnd_split[j][this_dom_pos];
		Cnnz += bnd_split[j][this_dom_pos+1] - in_split[j];
		
		// select the next node
		node = node->next;
		j++;
	}
	
	// allocate memory for F and C
	mtx_CRS_init( &P->A.F, n_bnd, n_in,  Fnnz, block );
	mtx_CRS_init( &P->A.C, n_bnd, n_bnd, Cnnz, block );
	P->A.F.rindx[0] = 0;
	P->A.C.rindx[0] = 0;

	// allocate memory for split matrices
	P->A.Eij = (Tmtx_CRS_ptr)malloc( sizeof(Tmtx_CRS)*n_neigh );
	for( i=0; i<n_neigh; i++ )
	{
		dom = P->backward.neighbours[i];
		if( dom!=this_dom )
		{
			mtx_CRS_init( P->A.Eij+i, n_bnd,  P->domains.n_bnd[dom], Eijnnz[i], block );
			P->A.Eij[i].rindx[0] = 0;
		}
		else
		{
			P->A.Eij[i].init = 0;
		}
	}
	
	// now cut-up and store the nodes in the matrices
	node = bnd.start;
	j = 0;
	while( node!=NULL )
	{				
		/* 
		 *	store the Eij rows for This node
		 */
		for( i=0; i<n_neigh; i++ )
		{
			dom = P->backward.neighbours[i];
			
			if( dom!=this_dom )
			{
				// determine the number of nz elements in This row of Eij[i]
				// then update the rindx for Eij[i] appropriately
				row_nnz = bnd_split[j][i+1] - bnd_split[j][i];
				P->A.Eij[i].rindx[j+1] = P->A.Eij[i].rindx[j] + row_nnz;
				
				// if there are nonzero elements in This row add them to the matrix
				if( row_nnz )
				{
					// copy the cindx values, making them local to the matrix Eij
					spot = P->domains.vtxdist[dom] + P->domains.n_in[dom];
					k = bnd_split[j][i];
					for( pos=P->A.Eij[i].rindx[j]; pos<P->A.Eij[i].rindx[j+1]; pos++, k++ )
						P->A.Eij[i].cindx[pos] = node->indx[k]-spot;
					
					// copy over the nonzero values
					if( block )
						memcpy( P->A.Eij[i].nz + (P->A.Eij[i].rindx[j] BLOCK_M_SHIFT), node->dat + (bnd_split[j][i] BLOCK_M_SHIFT), (row_nnz BLOCK_M_SHIFT)*sizeof(double) );
					else
						memcpy( P->A.Eij[i].nz + P->A.Eij[i].rindx[j], node->dat + bnd_split[j][i], row_nnz*sizeof(double) );
				}
			}
		}
		
		/* 
		 *  F and C blocks for This node
		 */
		
		// add row to F
		row_nnz = in_split[j] - bnd_split[j][this_dom_pos];
		P->A.F.rindx[j+1] = P->A.F.rindx[j] + row_nnz;
		if( row_nnz )
		{
			k = bnd_split[j][this_dom_pos];
			spot = P->domains.vtxdist[this_dom];
			for( pos=P->A.F.rindx[j]; pos<P->A.F.rindx[j+1]; pos++, k++ )
				P->A.F.cindx[pos] = node->indx[k]-spot;
			if( block )
				memcpy( P->A.F.nz + (P->A.F.rindx[j] BLOCK_M_SHIFT), node->dat + (bnd_split[j][this_dom_pos] BLOCK_M_SHIFT), (row_nnz BLOCK_M_SHIFT)*sizeof(double)  );
			else
				memcpy( P->A.F.nz + P->A.F.rindx[j], node->dat + bnd_split[j][this_dom_pos], row_nnz*sizeof(double)  );
		}
		
		// add row to C
		row_nnz = bnd_split[j][this_dom_pos+1] - in_split[j];
		P->A.C.rindx[j+1] = P->A.C.rindx[j] + row_nnz;
		if( row_nnz )
		{
			k = in_split[j];
			spot = P->domains.vtxdist[this_dom] + P->domains.n_in[this_dom];
			for( pos=P->A.C.rindx[j]; pos<P->A.C.rindx[j+1]; pos++, k++ )
				P->A.C.cindx[pos] = node->indx[k] - spot;
			if( block )
				memcpy( P->A.C.nz + (P->A.C.rindx[j] BLOCK_M_SHIFT), node->dat + (in_split[j] BLOCK_M_SHIFT), (row_nnz BLOCK_M_SHIFT)*sizeof(double)  );
			else
				memcpy( P->A.C.nz + P->A.C.rindx[j], node->dat + in_split[j], row_nnz*sizeof(double)  );
		}
		
		// increment line counter
		j++;
		node = node->next;
	}
	
	// free up working arrays
	free( in_split );
	for( k=0; k<n_bnd; k++ )
		free( bnd_split[k] );
	free( bnd_split );
	free( int_temp );
	free( Eijnnz );
	
	// free up memory used in This process
	fclose( fid );
}

/******************************************************************************************
*		mtx_CRS_dist_split_global()
*
*		take a distributed CRS matrix A and split it into the Schur form in 
*		the precondioner P. The matrix A is already in the decomposed form.
*
*		P has been initialised and A is initialised and valid.
*******************************************************************************************/
void mtx_CRS_dist_split_global( Tmtx_CRS_dist_ptr A, Tprecon_Schur_ptr P, Tdomain_ptr D )
{
	TMPI_dat_ptr This;
	int n_dom, this_dom, block, n_in, n_bnd, n_neigh, dom, dom_start, this_dom_pos;
	int i, j, k, pos, jj, spot, row_nnz;
	int Ennz, Bnnz, Fnnz, Cnnz, offset, row_start;
	int *in_split, *cx, *neighbours, *Eijnnz, **bnd_split;
	double *nz;
	
	// check that we are being passed initialised variables
	ASSERT( A->init );
	
	// copy the domain information to P
	domain_copy( D, &P->domains );
	
	// initialise variables
	This = &A->This;
	block = A->mtx.block;
	n_dom = This->n_proc;
	this_dom = This->this_proc;
	
	/*
	 *		Store the split of A
	 *
	 *		Done in two steps :
	 *			1.  all indices in nodes are changed to new order and sorted.
	 *				during This the number of nonzero entries in each of the split
	 *				matrices is calculated.
	 *			2.  memory is allocated for the split matrices and the node data is copied
	 *				into the splits.
	 *
	 */
	
	/* 
	 *		do the interior nodes first, these contribute to the matrices B and E
	 */
	
	dom_start = P->domains.vtxdist[this_dom];
	k = n_dom;
	k = P->domains.vtxdist[this_dom];
	n_in = P->domains.n_in[this_dom];
	n_bnd = P->domains.n_bnd[this_dom];
	n_neigh = 0;
	for( i=this_dom*n_dom; i<(this_dom+1)*n_dom; i++ )
	{
		// make sure that we count ourselves
		if( (i%n_dom)==this_dom )
			n_neigh++;
		else if( P->domains.map[i] )
			n_neigh++;
	}
	neighbours = (int*)malloc( sizeof(int)*n_neigh );
	for( pos=0, i=0; i<n_dom; i++ )
	{	 
		if( i==this_dom )
			neighbours[pos++]=i;
		else if( P->domains.map[i+n_dom*this_dom] )
			neighbours[pos++]=i;
	}
	
	Ennz = Bnnz = 0;
	in_split = (int *)malloc( sizeof(int)*n_in );
	
	// loop through the nodes
	for( j=0; j<n_in; j++ )
	{ 
		pos=A->mtx.rindx[j];
		while( pos<A->mtx.rindx[j+1] && A->mtx.cindx[pos]<n_in+dom_start )
			pos++;
		Bnnz += pos-A->mtx.rindx[j];
		Ennz += ( A->mtx.rindx[j+1] - pos);
		in_split[j] = pos;
	}
	
	/*
	 *  now make the matrices B and E
	 */
	
	// allocate  memory
	mtx_CRS_init( &P->A.B, n_in, n_in,  Bnnz, block );
	mtx_CRS_init( &P->A.E, n_in, n_bnd, Ennz, block );
	
	// add rows one at a time
	P->A.B.rindx[0] = 0;
	P->A.E.rindx[0] = 0;
	nz = A->mtx.nz;
	cx = A->mtx.cindx;
	for( j=0; j<n_in; j++ )
	{
		// add row to B
		row_start = A->mtx.rindx[j];
		k = in_split[j]-row_start;
		P->A.B.rindx[j+1] = P->A.B.rindx[j] + k;
		if( k )
		{
			memcpy( P->A.B.cindx + P->A.B.rindx[j], cx+row_start, k*sizeof(int) );
			offset = A->vtxdist[this_dom];
			for( i=P->A.B.rindx[j]; i<P->A.B.rindx[j+1]; i++ )
				P->A.B.cindx[i] -= offset;
			if( block )
				memcpy( P->A.B.nz + (P->A.B.rindx[j] BLOCK_M_SHIFT), nz+(row_start BLOCK_M_SHIFT), (k BLOCK_M_SHIFT)*sizeof(double)  );
			else
				memcpy( P->A.B.nz + P->A.B.rindx[j], nz+row_start, k*sizeof(double) );
		}
		
		// add row to E
		k = A->mtx.rindx[j+1] - in_split[j];
		P->A.E.rindx[j+1] = P->A.E.rindx[j] + k;
		if( k )
		{
			// make cindx values for E start at zero
			memcpy( P->A.E.cindx + P->A.E.rindx[j], cx + in_split[j], k*sizeof(int) );
			offset = A->vtxdist[this_dom]+n_in;
			for( i=P->A.E.rindx[j]; i<P->A.E.rindx[j+1]; i++ )
				P->A.E.cindx[i] -= offset;
			if( block )
				memcpy( P->A.E.nz + (P->A.E.rindx[j] BLOCK_M_SHIFT), nz + (in_split[j] BLOCK_M_SHIFT), (k BLOCK_M_SHIFT)*sizeof(double) );
			else
				memcpy( P->A.E.nz + P->A.E.rindx[j], nz + in_split[j], k*sizeof(double)  );
		}
	}
		
	// free up working arrays
	free( in_split );
	
	/* 
	 *		do the exterior nodes, these contribute to the matrices F, C and the Eij
	 */
	
	// allocate memory for temporary arrays
	in_split  = (int *)malloc( sizeof(int)*n_bnd );
	Eijnnz    = (int *)calloc( n_neigh, sizeof(int) );
	bnd_split = (int **)malloc( sizeof(int*)*n_bnd );
	for( k=0; k<n_bnd; k++ )
		bnd_split[k] = (int *)malloc( sizeof(int)*(n_neigh+1) );
	
	// intialise loop variables
	j = 0;
	this_dom_pos = 0;
	Fnnz = Cnnz = 0;
	while( neighbours[this_dom_pos] != this_dom )
		this_dom_pos++;
	
	// loop over nodes
	for( j=0; j<n_bnd; j++ )
	{		
		// determine the which elements belong to which split
		jj = j + n_in;
		pos = A->mtx.rindx[jj];
		for( i=0; i<n_neigh; i++ )
		{
			bnd_split[j][i]=pos;
			dom = neighbours[i];
			spot = P->domains.vtxdist[dom+1];
			while( pos<A->mtx.rindx[jj+1] && A->mtx.cindx[pos] < spot )
				pos++;
		}
		bnd_split[j][n_neigh] = pos;
		
		// work specifically on the diagonal split
		pos = bnd_split[j][this_dom_pos];
		while( pos<A->mtx.rindx[jj+1] && A->mtx.cindx[pos]-dom_start < n_in )
			pos++;
		in_split[j] = pos;
		
		// update the nnz counts
		for( i=0; i<n_neigh; i++ )
			Eijnnz[i] += bnd_split[j][i+1]-bnd_split[j][i];
		
		Fnnz += in_split[j] - bnd_split[j][this_dom_pos];
		Cnnz += bnd_split[j][this_dom_pos+1] - in_split[j];
	}
	
	// allocate memory for F and C
	mtx_CRS_init( &P->A.F, n_bnd, n_in,  Fnnz, block );
	mtx_CRS_init( &P->A.C, n_bnd, n_bnd, Cnnz, block );
	P->A.F.rindx[0] = 0;
	P->A.C.rindx[0] = 0;
	
	// allocate memory for split matrices
	P->A.Eij = (Tmtx_CRS_ptr)malloc( sizeof(Tmtx_CRS)*n_neigh );
	for( i=0; i<n_neigh; i++ )
	{
		dom = neighbours[i];
		if( dom!=this_dom )
		{
			mtx_CRS_init( P->A.Eij+i, n_bnd,  P->domains.n_bnd[dom], Eijnnz[i], block );
			P->A.Eij[i].rindx[0] = 0;
		}
		else
		{
			P->A.Eij[i].init = 0;
		}
	}
	
	// now cut-up and store the nodes in the matrices
	for( j=0; j<n_bnd; j++ )
	{		
		jj = j + n_in;
		
		/* 
		 *	store the Eij rows for This node
		 */
		for( i=0; i<n_neigh; i++ )
		{
			dom = neighbours[i];
			
			if( dom!=this_dom )
			{
				// determine the number of nz elements in This row of Eij[i]
				// then update the rindx for Eij[i] appropriately
				row_nnz = bnd_split[j][i+1] - bnd_split[j][i];
				P->A.Eij[i].rindx[j+1] = P->A.Eij[i].rindx[j] + row_nnz;
				
				// if there are nonzero elements in This row add them to the matrix
				if( row_nnz )
				{
					// copy the cindx values, making them local to the matrix Eij
					spot = P->domains.vtxdist[dom] + P->domains.n_in[dom];
					k = bnd_split[j][i];
					for( pos=P->A.Eij[i].rindx[j]; pos<P->A.Eij[i].rindx[j+1]; pos++, k++ )
						P->A.Eij[i].cindx[pos] = A->mtx.cindx[k]-spot;
					
					// copy over the nonzero values
					if( block )
						memcpy( P->A.Eij[i].nz + (P->A.Eij[i].rindx[j] BLOCK_M_SHIFT), A->mtx.nz + (bnd_split[j][i] BLOCK_M_SHIFT), (row_nnz BLOCK_M_SHIFT)*sizeof(double) );
					else
						memcpy( P->A.Eij[i].nz + P->A.Eij[i].rindx[j], A->mtx.nz + bnd_split[j][i], row_nnz*sizeof(double) );
				}
			}
		}
		
		/* 
		 *  F and C blocks for This node
		 */
		
		// add row to F
		row_nnz = in_split[j] - bnd_split[j][this_dom_pos];
		P->A.F.rindx[j+1] = P->A.F.rindx[j] + row_nnz;
		if( row_nnz )
		{
			k = bnd_split[j][this_dom_pos];
			spot = P->domains.vtxdist[this_dom];
			for( pos=P->A.F.rindx[j]; pos<P->A.F.rindx[j+1]; pos++, k++ )
				P->A.F.cindx[pos] = A->mtx.cindx[k]-spot;
			if( block )
				memcpy( P->A.F.nz + (P->A.F.rindx[j] BLOCK_M_SHIFT), A->mtx.nz + (bnd_split[j][this_dom_pos] BLOCK_M_SHIFT), (row_nnz BLOCK_M_SHIFT)*sizeof(double)  );
			else
				memcpy( P->A.F.nz + P->A.F.rindx[j], A->mtx.nz + bnd_split[j][this_dom_pos], row_nnz*sizeof(double)  );
		}
		
		// add row to C
		row_nnz = bnd_split[j][this_dom_pos+1] - in_split[j];
		P->A.C.rindx[j+1] = P->A.C.rindx[j] + row_nnz;
		if( row_nnz )
		{
			k = in_split[j];
			spot = P->domains.vtxdist[this_dom] + P->domains.n_in[this_dom];
			for( pos=P->A.C.rindx[j]; pos<P->A.C.rindx[j+1]; pos++, k++ )
				P->A.C.cindx[pos] = A->mtx.cindx[k] - spot;
			if( block )
				memcpy( P->A.C.nz + (P->A.C.rindx[j] BLOCK_M_SHIFT), A->mtx.nz + (in_split[j] BLOCK_M_SHIFT), (row_nnz BLOCK_M_SHIFT)*sizeof(double)  );
			else
				memcpy( P->A.C.nz + P->A.C.rindx[j], A->mtx.nz + in_split[j], row_nnz*sizeof(double)  );
		}
	}
	
	{
		char f[100];
		sprintf( f, "test_B_%d.mtx", this_dom );
		mtx_CRS_output_matlab(  &P->A.B , f );
		sprintf( f, "test_C_%d.mtx", this_dom );
		mtx_CRS_output_matlab(  &P->A.C , f );
		sprintf( f, "test_E_%d.mtx", this_dom );
		mtx_CRS_output_matlab(  &P->A.E , f );
		sprintf( f, "test_F_%d.mtx", this_dom );
		mtx_CRS_output_matlab(  &P->A.F , f );
	}

	// save the list of neighbours
	P->A.neighbours = neighbours;
	P->n_bnd = n_bnd;
	P->n_neigh = n_neigh;
	P->n_in = n_in;
	P->n_local = n_in + n_bnd;
	
	// free up working arrays
	free( in_split );
	for( k=0; k<n_bnd; k++ )
		free( bnd_split[k] );
	free( bnd_split );
	free( Eijnnz );
}

/******************************************************************************************
*	schur_form_local()
*	
*   create a local schur compliment matrix. P is the preconditioner of the matrix B.
*
*   NOT FOR BLOCK YET
******************************************************************************************/
void schur_form_local_( Tprecon_ptr P, Tmtx_CRS_ptr E, Tmtx_CRS_ptr F, Tmtx_CRS_ptr C, Tmtx_CCS_ptr S )
{
	int *Ecolperm, *Ccolperm, vtxdist[2], *Crindx, *Erindx;
	long long *order;
	int n_in, n_bnd;
	int i, col, pos, Cpos, sp_pos;
	double *s;
	Tvec_dist x, y;
	
	// does  not support blocks yet
	ASSERT_MSG( !E->block, "schur_form_local() : This routine needs to be updated to deal with blocks." );
	
	/*
	 *		setup parameters
	 */
	
	// find the number of internal and boundary nodes
	n_in = E->nrows;
	n_bnd = E->ncols;
	
	// allocate memory
	vtxdist[0] = 0;
	vtxdist[1] = n_in;
	vec_dist_init( &x, &P->This, n_in, E->block, vtxdist );
	vec_dist_init( &y, &P->This, n_in, E->block, vtxdist );
	mtx_CCS_init( S, n_bnd, n_bnd, 10*n_bnd, E->block );
	
	/* 
	 *		get profile data from E and C
	 */
	
	// allocate memory for work arrays
	Erindx   = (int *)malloc( sizeof(int)*E->nnz );
	Crindx   = (int *)malloc( sizeof(int)*C->nnz );
	Ecolperm = (int *)malloc( sizeof(int)*E->nnz );
	Ccolperm = (int *)malloc( sizeof(int)*C->nnz );
	order    = (long long *)malloc( sizeof(long long)*E->nnz );

	// determine the CCS permutation array for E
	index_CRS_to_CCS_findperm( E->cindx, E->rindx, Ecolperm, Erindx, order, n_in, E->nnz );
	
	// reallocate work array for C
	order = (long long *)realloc( order, sizeof(long long)*C->nnz );
	
	// determine the CCS permutation array for C
	index_CRS_to_CCS_findperm( C->cindx, C->rindx, Ccolperm, Crindx, order, n_bnd, C->nnz );
	
	// free up memory
	free( order );
	
	/*
	 *		create the columns of S one-at-a-time and store in place
	 */
	S->cindx[0] = 0;
	for( Cpos=0, pos=0, col=0; col<n_bnd; col++ )
	{	
		// check if we need to allocate more memory in S
		if( (S->nnz - sp_pos)<n_bnd )
		{
			int new_size;
			
			// allocate space for 1.2 times more nz entries per column than are
			// currently being used
			new_size = ceil( (double)col/(double)sp_pos*1.2 )*n_bnd;
			
			S->rindx = (int *)   realloc( S->rindx, sizeof(int)*new_size );
			S->nz    = (double *)realloc( S->nz,    sizeof(double)*new_size );
		}
		
		// form the dense column of E
		for( i=0; i<n_in; i++ )
			x.dat[i] = 0.;
		while( pos<E->nnz && E->cindx[Ecolperm[pos]]==col )
		{
			x.dat[Erindx[pos]] = E->nz[Ecolperm[pos]];
			pos++;
		}
		
		// apply preconditioner to it, storing in y
		precon_apply( P, &x, &y );
		
		// multiply F on LHS  of preconditioned column
		// s = -F*y;
		s = S->nz + S->cindx[col];
		mtx_CRS_gemv( F, y.dat, s, -1., 0., 'n' );
		
		// adjust with the relevant values from C
		while( Cpos<C->nnz && C->cindx[Ccolperm[Cpos]]==col )
		{
			//printf( "\t\t\tCpos = %d and colperm = %d and the nz is %g and our pos in s is %d\n", Cpos, Ccolperm[Cpos], C->nz[Ccolperm[Cpos]], Crindx[Cpos] );
			s[Crindx[Cpos]] += C->nz[Ccolperm[Cpos]];
			Cpos++;
		}		
		
		// store
		sp_pos = S->cindx[col];
		for( i=0; i<n_bnd; i++ )
		{
			if( abs(s[i]) )
			{
				S->nz[sp_pos] = s[i];
				S->rindx[sp_pos++] = i;
			}
		}
		S->cindx[col+1] = sp_pos;
	}
	
	// realloc to make the entries in S a snug fit
	S->nnz   = sp_pos;
	S->rindx = (int *)   realloc( S->rindx, sizeof(int)*sp_pos );
	S->nz    = (double *)realloc( S->nz,    sizeof(double)*sp_pos );
	
	/*
	 *		clean up
	 */
	vec_dist_free( &x );
	vec_dist_free( &y );
	free( Erindx );
	free( Crindx );
	free( Ecolperm );
	free( Ccolperm );
}

/******************************************************************************************
*	schur_form_local()
*	
*   create a local schur compliment matrix. P is the preconditioner of the matrix B.
******************************************************************************************/
void schur_form_local( Tprecon_ptr P, Tmtx_CRS_ptr E, Tmtx_CRS_ptr F, Tmtx_CRS_ptr C, Tmtx_CCS_ptr S )
{
	int vtxdist[2];
	Tmtx_CCS E_CCS, C_CCS;
	int n_in, n_bnd;
	int i, col, pos, sp_pos;
	double *s;

	Tvec_dist x, y;
		
	/*
	 *		setup parameters
	 */
	
	// find the number of internal and boundary nodes
	n_in = E->nrows;
	n_bnd = E->ncols;
	
	// allocate memory
	vtxdist[0] = 0;
	vtxdist[1] = n_in;
	vec_dist_init( &x, &P->This, n_in, E->block, vtxdist );
	vec_dist_init( &y, &P->This, n_in, E->block, vtxdist );
	mtx_CCS_init( S, n_bnd, n_bnd, 10*n_bnd, E->block );
	
	/* 
     *		Form CCS versions of E and C
	 */
	
	mtx_CRS_to_CCS( C, &C_CCS );
	mtx_CRS_to_CCS( E, &E_CCS );
	
	/*
	 *		create the columns of S one-at-a-time and store in place
	 */
	S->cindx[0] = 0;
	for( col=0; col<n_bnd; col++ )
	{			
		// check if we need to allocate more memory in S
		while( (S->nnz - sp_pos)<n_bnd )
		{
			int new_size;
			
			// allocate space for 1.2 times more nz entries per column than are
			// currently being used
			new_size = ceil( (double)sp_pos/(double)col*1.2 )*n_bnd;
			
			S->rindx = (int *)   realloc( S->rindx, sizeof(int)*new_size );
			S->nz    = (double *)realloc( S->nz,    sizeof(double)*new_size );
		}
		
		// form the dense column of E
		for( i=0; i<n_in; i++ )
			x.dat[i] = 0.;
		for( pos=E_CCS.cindx[col]; pos<E_CCS.cindx[col+1]; pos++ )
		{
			x.dat[E_CCS.rindx[pos]] = E->nz[pos];
		}
		
		// apply preconditioner to it, storing in y
		precon_apply( P, &x, &y );
		
		// multiply F on LHS  of preconditioned column
		// s = -F*y;
		s = S->nz + S->cindx[col];
		mtx_CRS_gemv( F, y.dat, s, -1., 0., 'n' );
			
		// adjust with the relevant values from C
		for( pos=C_CCS.cindx[col]; pos<C_CCS.cindx[col+1]; pos++ )
		{
			s[C_CCS.rindx[pos]] += C->nz[pos];
		}
		
		// store
		sp_pos = S->cindx[col];
		for( i=0; i<n_bnd; i++ )
		{
			if( s[i] )
			{
				S->nz[sp_pos] = s[i];
				S->rindx[sp_pos++] = i;
			}
		}
		S->cindx[col+1] = sp_pos;
	}
	
	// realloc to make the entries in S a snug fit
	S->nnz   = sp_pos;
	S->rindx = (int *)   realloc( S->rindx, sizeof(int)*sp_pos );
	S->nz    = (double *)realloc( S->nz,    sizeof(double)*sp_pos );
	
	/*
	 *		clean up
	 */
	vec_dist_free( &x );
	vec_dist_free( &y );
	mtx_CCS_free( &C_CCS );
	mtx_CCS_free( &E_CCS );
}

/******************************************************************************************
*	schur_form_local_block()
*	
*   create a local schur compliment matrix. P is the preconditioner of the matrix B.
******************************************************************************************/
void __schur_form_local_block( Tprecon_ptr P, Tmtx_CRS_ptr E, Tmtx_CRS_ptr F, Tmtx_CRS_ptr C, Tmtx_CCS_ptr S )
{
	int vtxdist[2];
	Tmtx_CCS E_CCS, C_CCS;
	int n_in, n_bnd;
	int i, col, pos, sp_pos, *sparsity, *sparsity_p, nz_count;
	double *s, *edat, *ydat, *sdat, *copy_from, *copy_to;
	Tvec_dist x, y;
	
	/*
	 *		setup parameters
	 */
	
	// find the number of internal and boundary nodes
	n_in = E->nrows;
	n_bnd = E->ncols;
	
	// allocate memory
	vtxdist[0] = 0;
	vtxdist[1] = n_in;
	vec_dist_init( &x, &P->This, n_in, E->block, vtxdist );
	vec_dist_init( &y, &P->This, n_in, E->block, vtxdist );
	free( x.dat );
	free( y.dat );
	mtx_CCS_init( S, n_bnd, n_bnd, 10*n_bnd, E->block );
	edat = (double*)malloc( (sizeof(double)*n_in) BLOCK_M_SHIFT );
	ydat = (double*)malloc( (sizeof(double)*n_in) BLOCK_M_SHIFT );
	sdat = (double*)malloc( (sizeof(double)*n_bnd) BLOCK_M_SHIFT );
	sparsity = (int*)malloc( sizeof(int)*(n_bnd+n_in) );

	
	/* 
	 *		Form CCS versions of E and C
	 */
		
	mtx_CRS_to_CCS( C, &C_CCS );
	mtx_CRS_to_CCS( E, &E_CCS );
	
	/*
	 *		create the columns of S one-at-a-time and store in place
	 */
	S->cindx[0] = 0;
	for( col=0; col<n_bnd; col++ )
	{		
		//printf( "col %d\n", col );
		
		// check if we need to allocate more memory in S
		while( (S->nnz - sp_pos)<n_bnd )
		{
			int new_size;
			
			new_size = ceil( (double)sp_pos/(double)col*1.2 )*n_bnd;
			
			S->rindx = (int *)   realloc( S->rindx, sizeof(int)*new_size );
			S->nz    = (double *)realloc( S->nz,    (sizeof(double)*new_size) BLOCK_M_SHIFT );
			S->nnz = new_size;
		}
		
		// form the dense columns of E
		mtx_CCS_column_unpack_block( &E_CCS, edat, col );
		
		// apply preconditioner to it, storing in y
		for( i=0; i<BLOCK_SIZE; i++ )
		{
			x.dat = edat + i*n_in*BLOCK_SIZE;
			y.dat = ydat + i*n_in*BLOCK_SIZE;
			precon_apply( P, &x, &y );
		}
		
		// multiply F on LHS  of preconditioned column
		// s = -F*y;
		for( i=0; i<BLOCK_SIZE; i++ )
		{
			s = sdat + i*n_bnd*BLOCK_SIZE;
			y.dat = ydat + i*n_in*BLOCK_SIZE;
			mtx_CRS_gemv( F, y.dat, s, -1., 0., 'n' );
		}
		
		// adjust with the relevant values from s = C - s =	C - F*inv(B)*E	
		mtx_CCS_column_add_unpacked_block( &C_CCS, sdat, col, 1., 1. );
		
		// determine the sparsity pattern of the schur (block) columm
		for( i=0; i<n_bnd; i++ )
			sparsity[i] = 0;
		for( i=0; i<BLOCK_SIZE; i++ )
		{
			s = sdat + i*n_bnd*BLOCK_SIZE;
			for( pos=0; pos<n_bnd; pos++, s+=BLOCK_SIZE )
			{
				if( BLOCK_V_ISNZ( s ) )
					sparsity[pos] = 1;
			}
		}
		sparsity_p = S->rindx + S->cindx[col];
		for( i=0, nz_count=0; i<n_bnd; i++ )
			if( sparsity[i] )
				sparsity_p[nz_count++] = i;

		// store the new (block) column of the Schur comliment
		sp_pos = S->cindx[col];
		for( i=0; i<BLOCK_SIZE; i++, s+=n_bnd )
		{
			s = sdat + i*n_bnd*BLOCK_SIZE;
			copy_to = S->nz + (sp_pos BLOCK_M_SHIFT) + (i BLOCK_V_SHIFT);
			for( pos=0; pos<nz_count; pos++, copy_to+=(BLOCK_SIZE*BLOCK_SIZE) )
			{
				copy_from = s + (sparsity_p[pos] BLOCK_V_SHIFT);
				BLOCK_V_COPY( copy_from, copy_to );
			}
		}
		
		S->cindx[col+1] = S->cindx[col] + nz_count;
	}
		
	// realloc to make the entries in S a snug fit
	S->nnz   = S->cindx[col];
	S->rindx = (int *)   realloc( S->rindx, sizeof(int)*S->nnz  );
	S->nz    = (double *)realloc( S->nz,    (sizeof(double)*S->nnz) BLOCK_M_SHIFT );

	/*
	 *		clean up
	 */
	x.dat = NULL;
	y.dat = NULL;
	vec_dist_free( &x );
	vec_dist_free( &y );
	mtx_CCS_free( &C_CCS );
	mtx_CCS_free( &E_CCS );
 	free( edat );  
 	free( ydat );  
 	free( sdat );
	free( sparsity );
}

/******************************************************************************************
*	schur_form_local_block()
*	
*   create a local schur compliment matrix. P is the preconditioner of the matrix B.
******************************************************************************************/
void schur_form_local_block( Tprecon_ptr P, Tmtx_CRS_ptr E, Tmtx_CRS_ptr F, Tmtx_CRS_ptr C, Tmtx_CCS_ptr S )
{
	int COL_FILL = 5, DROP_TOL = 0;
	int vtxdist[2];
	Tmtx_CCS E_CCS, C_CCS;
	int n_in, n_bnd;
	int i, col, pos, sp_pos, nz_count;
	double *s, *edat, *ydat, *sdat, *copy_from, *copy_to, *drop_temp;
	Tvec_dist x, y;	
	int rank;
	
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	
	/*
	 *		setup parameters
	 */
	
	// find the number of internal and boundary nodes
	n_in = E->nrows;
	n_bnd = E->ncols;
	
	// make sure that COL_FILL isn't too large
	COL_FILL = (COL_FILL<n_bnd) ? COL_FILL : n_bnd;
	
	// allocate memory
	vtxdist[0] = 0;
	vtxdist[1] = n_in;
	vec_dist_init( &x, &P->This, n_in, E->block, vtxdist );
	vec_dist_init( &y, &P->This, n_in, E->block, vtxdist );
	free( x.dat );
	free( y.dat );
	mtx_CCS_init( S, n_bnd, n_bnd, COL_FILL*n_bnd + n_bnd, E->block );
	edat = (double*)malloc( (sizeof(double)*n_in) BLOCK_M_SHIFT );
	ydat = (double*)malloc( (sizeof(double)*n_in) BLOCK_M_SHIFT );
	sdat = (double*)malloc( (sizeof(double)*n_bnd) BLOCK_M_SHIFT );
	drop_temp = (double*)malloc( sizeof(double)*n_bnd  );
	
	
	/* 
	 *		Form CCS versions of E and C
	 */
	
	mtx_CRS_to_CCS( C, &C_CCS );
	mtx_CRS_to_CCS( E, &E_CCS );
	
	/*
	 *		create the columns of S one-at-a-time and store in place
	 */
	S->cindx[0] = 0;
	for( col=0; col<n_bnd; col++ )
	{		
		
		//printf( "P%d\t:\tcol\t%d/%d\n", rank, col, n_bnd );
		
		// form the dense columns of E
		mtx_CCS_column_unpack_block( &E_CCS, edat, col );
		
		// apply preconditioner to it, storing in y
		for( i=0; i<BLOCK_SIZE; i++ )
		{
			x.dat = edat + i*n_in*BLOCK_SIZE;
			y.dat = ydat + i*n_in*BLOCK_SIZE;
			precon_apply( P, &x, &y );
		}
		
		// multiply F on LHS  of preconditioned column
		// s = -F*y;
		for( i=0; i<BLOCK_SIZE; i++ )
		{
			s = sdat + i*n_bnd*BLOCK_SIZE;
			y.dat = ydat + i*n_in*BLOCK_SIZE;
			mtx_CRS_gemv( F, y.dat, s, -1., 0., 'n' );
		}
		
		// adjust with the relevant values from s = C - s =	C - F*inv(B)*E	
		mtx_CCS_column_add_unpacked_block( &C_CCS, sdat, col, 1., 1. );
		
		// apply dropping to the column
		nz_count = vec_drop_block( sdat, S->rindx+S->cindx[col], n_bnd, COL_FILL, col, DROP_TOL, drop_temp );
		
		// store the new (block) column of the Schur comliment
		sp_pos = S->cindx[col];
		for( i=0; i<BLOCK_SIZE; i++, s+=n_bnd )
		{
			copy_from = sdat + i*nz_count*BLOCK_SIZE;
			copy_to = S->nz + (sp_pos BLOCK_M_SHIFT) + (i BLOCK_V_SHIFT);

			for( pos=0; pos<nz_count; pos++, copy_to+=(BLOCK_SIZE*BLOCK_SIZE), copy_from+=BLOCK_SIZE )
				BLOCK_V_COPY( copy_from, copy_to );
		}
		
		S->cindx[col+1] = S->cindx[col] + nz_count;
	}
	
	// realloc to make the entries in S a snug fit
	S->nnz   = S->cindx[col];
	S->rindx = (int *)   realloc( S->rindx, sizeof(int)*S->nnz  );
	S->nz    = (double *)realloc( S->nz,    (sizeof(double)*S->nnz) BLOCK_M_SHIFT );
	
	/*
	 *		clean up
	 */
	x.dat = NULL;
	y.dat = NULL;
	vec_dist_free( &x );
	vec_dist_free( &y );
	mtx_CCS_free( &C_CCS );
	mtx_CCS_free( &E_CCS );
 	free( edat );  
 	free( ydat );  
 	free( sdat );
	free( drop_temp );
}


/******************************************************************************************
*	schur_form_global()
*	
*   create the local chunk of the global schur compliment matrix.
*
*   P is the schur compliment preconditioner to which the schur chunk is to be added.
*
*   This is a bit fiddly, but there isn't really very much that can be done about it.
******************************************************************************************/
void schur_form_global( Tprecon_Schur_ptr P )
{
	int nnz, n_neigh, Sdim, n_bnd, this_dom, i, dom_i, pos;
	int dstart, len, n_dom, dom, row, block;
	Tmtx_CRS_ptr Ep;
	int *cindx, *Ecindx, *neighbours;
	double *nz;
	FILE *fid;
	char fname[30];
	
	/*
	 *		setup
	 */
		
	sprintf( fname, "schur_global_P%d.txt", P->This.this_proc );
	fid = fopen( fname, "w" );
	this_dom = P->This.this_proc;
	n_dom = P->This.n_proc;
	
	// block?
	block = P->Slocal.block;
	
	// how many neighbouring domains does This domain have?
	n_neigh = P->n_neigh;
	
	// how many boundary nodes in This domain
	n_bnd = P->n_bnd;
	
	fprintf( fid, "I have %d boundary nodes and I have %d neighbour domains\n\n", n_bnd, n_neigh-1 );
	
	// determine the total number of boundary nodes in the global domain
	Sdim = 0;
	for( i=0; i<n_dom; i++ )
		Sdim += P->domains.n_bnd[i];
	
	// neighbours points to the list of neighbours of This domain
	neighbours = P->A.neighbours;
	
	/*
	 *		augment the matrices
	 */
	
	// find total number of nz elements
	nnz = P->Slocal.nnz;
	for( i=0; i<n_neigh; i++ )
	{
		if( neighbours[i]!=this_dom )
		{
			fprintf( fid, "neighbour %d of %d contributes %d nz values\n\n", i, n_neigh-1, P->A.Eij[i].nnz );
			nnz += P->A.Eij[i].nnz;
		}
	}
	
	// initialise the global Schur
	mtx_CRS_dist_init( &P->S, n_bnd, Sdim, nnz, P->A.B.block, &P->This );
	fprintf( fid, "Initialsed Schur global chunk of dimension %dX%d with %d nz on P%d\n\n", n_bnd, Sdim, nnz, P->S.This.this_proc );
	BMPI_print( fid, &P->S.This );
	
	// create the schur, one row at a time
	cindx = P->S.mtx.cindx;
	nz    = P->S.mtx.nz;
	P->S.mtx.rindx[0] = 0;

	
	fprintf( fid, "Starting augmentation\n\n" );
	for( row=0, pos=0; row<n_bnd; row++ )
	{
		dom_i = 0;
		Ep = P->A.Eij;
		
		fprintf( fid, "\trow %d\n", row );
		
		// add parts from domains 0..this_dom-1
		while( (dom=neighbours[dom_i])<this_dom )
		{
			Ecindx = Ep->cindx + Ep->rindx[row];
			dstart = P->S.vtxdist[ dom ];
			len = Ep->rindx[row+1] - Ep->rindx[row];
			fprintf( fid, "\t\tadding contribution of %d elements from domain %d\n", len, dom );
			if( len )
			{
				if( block )
					memcpy( nz, Ep->nz + (Ep->rindx[row] BLOCK_M_SHIFT), sizeof(double)*(len BLOCK_M_SHIFT) );	
				else
					memcpy( nz, Ep->nz + Ep->rindx[row], sizeof(double)*len );
				for( i=0; i<len; i++ )
					cindx[i] = Ecindx[i] + dstart;
			}
			
			if( block )
				nz += len BLOCK_M_SHIFT;
			else
				nz += len;
			cindx += len;
			pos += len;
			Ep++;
			dom_i++;
		}
		
		// add local schur part
		dom = this_dom;
		Ecindx = P->Slocal.cindx + P->Slocal.rindx[row];
		dstart = P->S.vtxdist[ dom ];
		len = P->Slocal.rindx[row+1] - P->Slocal.rindx[row];
		fprintf( fid, "\t\tadding contribution of %d elements from domain %d\n", len, dom );
		if( len )
		{
			if( block ) 
				memcpy( nz, P->Slocal.nz + (P->Slocal.rindx[row] BLOCK_M_SHIFT), sizeof(double)*(len BLOCK_M_SHIFT) );
			else
				memcpy( nz, P->Slocal.nz + P->Slocal.rindx[row], sizeof(double)*len );
			for( i=0; i<len; i++ )
				cindx[i] = Ecindx[i] + dstart;
		}
		
		Ep++;
		if( block )
			nz += len BLOCK_M_SHIFT;
		else
			nz += len;
		cindx += len;
		pos   += len;				
		dom_i++;
		
		// add parts from domains this_dom+1..n_dom-1
		while( dom_i<n_neigh )
		{
			dom = neighbours[dom_i];
			Ecindx = Ep->cindx + Ep->rindx[row];
			dstart = P->S.vtxdist[ dom ];
			len = Ep->rindx[row+1] - Ep->rindx[row];
			fprintf( fid, "\t\tadding contribution of %d elements from domain %d\n", len, dom );
			if( len )
			{
				if( block )
					memcpy( nz, Ep->nz + (Ep->rindx[row]  BLOCK_M_SHIFT), sizeof(double)*(len  BLOCK_M_SHIFT));
				else
					memcpy( nz, Ep->nz + Ep->rindx[row], sizeof(double)*len );
				for( i=0; i<len; i++ )
					cindx[i] = Ecindx[i] + dstart;
			}
			
			if( block )
				nz += len BLOCK_M_SHIFT;
			else
				nz += len;
			cindx += len;
			pos += len;
			Ep++;
			dom_i++;
		}
		
		P->S.mtx.rindx[row+1] = pos;
	}
	
	fprintf( fid, "finished augmentation\n\n" );
	
	fprintf( fid, "the global Schur is thus : \n\n" );
	mtx_CRS_print( fid, &P->S.mtx );
	{
		Tmtx D;
		
		mtx_CRS_to_mtx( &P->S.mtx, &D );
		mtx_print( fid, &D );
		mtx_free( &D );
	}
	
	fclose( fid );
}

/******************************************************************************************
*	precon_schur()
*
*   form a schur compliment preconditioner in P for the matrix A
*
*   NOT FOR BLOCK YET
*
******************************************************************************************/
void precon_schur( Tmtx_CRS_dist_ptr A, Tprecon_Schur_ptr P, Tschur_param_ptr params, int level )
{
	Tmtx_CCS S_CCS_tmp;
	int precon_local;
	
	/*
	 *		Initialise the preconditioner
	 */
	precon_local = PRECON_JACOBI;
	precon_Schur_init( P, A );
	P->initial_decomp = 1;
	P->level = level;
	
	/*
	 *		Form the preconditioner for the Schur compliment if we are on level 0
	 */
	if( !level )
	{
		char fname[100];
		
		// do some diagnostics
		if( !P->This.this_proc )
			printf( "\t\tFinal Level... forming preconditioner for P->S\n" );
		mtx_CRS_dist_copy( A, &P->S );
		sprintf( fname, "S_%d.mtx", P->This.this_proc );
		mtx_CRS_output_matlab( &P->S.mtx, fname );
		
		// form global preconditioner for the final Schur compliment
		precon_init( &P->MS, PRECON_JACOBI, 1, &P->This );
		precon_jacobi( A, P->MS.preconditioner );
		return;
	}
	
	/*
	 *		Otherwise we need to perform another recursion
	 */
	else
	{
		if( !P->This.this_proc )
			printf( "\t%d splitting A...\n", level );
		
		//  distribute the matrix amongst the nodes, does This if needed or not
		mtx_CRS_dist_split( A, P );
		
		if( !P->This.this_proc )
			printf( "\t%d Creating Local Bi precon...\n", level );
		
		// form preconditioner for the local B matrix
		if( precon_local == PRECON_JACOBI )
		{
			Tmtx_CRS_dist B_dist;
			//char fname[100];
			
			if( !P->This.this_proc )
				printf( "\t\t%d Jacobi...\n", level );
			
			precon_init( &P->MB, PRECON_JACOBI, 1, P->This.sub_comm );
			mtx_CRS_distribute( &P->A.B, &B_dist, P->This.sub_comm, 0 );
			precon_jacobi( &B_dist, P->MB.preconditioner );
			mtx_CRS_dist_free( &B_dist );
		}
		else if( precon_local == PRECON_ILU0)
		{
			if( !P->This.this_proc )
				printf( "\t\t%d ILU0...\n", level );
			
			precon_init( &P->MB, PRECON_ILU0, 1, P->This.sub_comm );
			precon_ILU0( &P->A.B, P->MB.preconditioner );
		}
		else
		{
			fprintf( stderr, "ERROR : preconditioner type " ); precon_print_name( precon_local ); fprintf( stderr, " not supported as an inner preconditioner\n\n" );
			MPI_Finalize();
			exit(1);
		}
		
		// form the schur compliment
		if( A->mtx.block )
			schur_form_local_block( &P->MB, &P->A.E, &P->A.F, &P->A.C, &S_CCS_tmp );
		else
			schur_form_local( &P->MB, &P->A.E, &P->A.F, &P->A.C, &S_CCS_tmp );
		mtx_CCS_to_CRS( &S_CCS_tmp, &P->Slocal );

		
		if( !P->This.this_proc )
			printf( "\t%d Forming global Schur...\n", level );
		
		// form the global Schur compliment
		schur_form_global( P );
		
		if( !P->This.this_proc )
			printf( "\t%d Going down another level...\n", level );
		
		// precondition the global Schur compliment
		MPI_Barrier( P->This.comm );
		precon_init( &P->MS, PRECON_SCHUR_SPLIT, 1, &P->This );
		precon_schur( &P->S, (Tprecon_Schur_ptr)P->MS.preconditioner, params, level-1 );
	}
}

/******************************************************************************************
*	precon_schur_global()
*
*   form a schur compliment preconditioner in P for the matrix A where A has already been
*	decomposed and distributed. Distribution data is in dom
*
*   NOT FOR BLOCK YET
*
******************************************************************************************/
void precon_schur_global( Tmtx_CRS_dist_ptr A, Tprecon_Schur_ptr P, Tschur_param_ptr params, int level, Tdomain_ptr dom )
{
	Tmtx_CCS S_CCS_tmp;
	int precon_local;
	
	/*
	 *		Initialise the preconditioner
	 */
	precon_local = PRECON_JACOBI;
	precon_local = PRECON_ILU0;
	precon_Schur_init( P, A );
	P->level = level;
	P->initial_decomp = 1;
	domain_copy( dom, &P->domains);
	
	/*
	 *		Form the preconditioner for the Schur compliment if we are on level 0
	 */
	if( !level )
	{
		char fname[100];
		
		// do some diagnostics
		if( !P->This.this_proc )
			printf( "\t\tFinal Level... forming preconditioner for P->S\n" );
		mtx_CRS_dist_copy( A, &P->S );
		sprintf( fname, "S_%d.mtx", P->This.this_proc );
		mtx_CRS_output( &P->S.mtx, fname );
		
		// form global preconditioner for the final Schur compliment
		precon_init( &P->MS, PRECON_JACOBI, 1, &P->This );
		precon_jacobi( A, P->MS.preconditioner );
		return;
	}
	
	/*
	 *		Otherwise we need to perform another recursion
	 */
	else
	{
		if( !P->This.this_proc )
			printf( "\t%d splitting A...\n", level );
		
		//  distribute the matrix amongst the nodes, does This if needed or not
		mtx_CRS_dist_split_global( A, P, dom );
		
		if( !P->This.this_proc )
			printf( "\t%d Creating Local Bi precon...\n", level );
		
		// form preconditioner for the local B matrix
		if( precon_local == PRECON_JACOBI )
		{
			Tmtx_CRS_dist B_dist;
			
			if( !P->This.this_proc )
				printf( "\t\t%d Jacobi...\n", level );
			
			precon_init( &P->MB, PRECON_JACOBI, 1, P->This.sub_comm );
			mtx_CRS_distribute( &P->A.B, &B_dist, P->This.sub_comm, 0 );
			precon_jacobi( &B_dist, P->MB.preconditioner );
			mtx_CRS_dist_free( &B_dist );
		}
		else if( precon_local == PRECON_ILU0)
		{
			if( !P->This.this_proc )
				printf( "\t\t%d ILU0...\n", level );
			
			precon_init( &P->MB, PRECON_ILU0, 1, P->This.sub_comm );
			precon_ILU0( &P->A.B, P->MB.preconditioner );
		}
		else
		{
			fprintf( stderr, "ERROR : preconditioner type " ); precon_print_name( precon_local ); fprintf( stderr, " not supported as an inner preconditioner\n\n" );
			MPI_Finalize();
			exit(1);
		}
		
		if( !P->This.this_proc )
			printf( "\t%d Forming local Schur...\n", level );
		
		// form the schur compliment
		if( A->mtx.block )
			schur_form_local_block( &P->MB, &P->A.E, &P->A.F, &P->A.C, &S_CCS_tmp );
		else
			schur_form_local( &P->MB, &P->A.E, &P->A.F, &P->A.C, &S_CCS_tmp );
		
		// convert the local schur compliment to CRS fomrat
		P->Slocal.init = 0;
		mtx_CCS_to_CRS( &S_CCS_tmp, &P->Slocal );
		mtx_CCS_free( &S_CCS_tmp );
				
		// form the global Schur compliment
		if( !P->This.this_proc )
			printf( "\t%d Forming global Schur...\n", level );
		schur_form_global( P );
		
		if( !P->This.this_proc )
			printf( "\t%d Going down another level...\n", level );
		
		// precondition the global Schur compliment
		MPI_Barrier( P->This.comm );
		precon_init( &P->MS, PRECON_SCHUR_SPLIT, 1, &P->This );
		precon_schur( &P->S, (Tprecon_Schur_ptr)P->MS.preconditioner, params, level-1 );
	}
}

/*************************************************************
 *		precon_schur_apply()
 *
 *		WARNING : does not support block matrices yet
 *
 *************************************************************/
void precon_schur_apply( Tprecon_Schur_ptr P, Tvec_dist_ptr x, Tvec_dist_ptr y )
{
	Tvec_dist xl, xL, yB, v, PxL, yL;
	int k, n_dom, this_dom, i, block;
	int *vtxdist_in, *vtxdist_bnd;
	double *tmp_double, *dat;
	MPI_Request *request;
	MPI_Status *status;
	char fname[100];
	
	n_dom    = P->This.n_proc;
	this_dom = P->This.this_proc;
	block = P->Slocal.block;
	xl.init = xL.init = yB.init = v.init = PxL.init = yL.init = 0;
	
	/*
	 *		if we are at the bottom level then solve the low level GMRES problem
	 */	
	if( !P->level )
	{		
		Tgmres gmres_stats;
		
		// setup the GMRES parameters
		GMRES_setup_schur( &gmres_stats );
		
		// setup the diagnostic output file
		if( 0 ) //!this_dom )
		{
			// open output file
			sprintf( fname, "precon_apply_inner_%d.txt", this_dom );
			gmres_stats.diagnostic = fopen( fname, "w" );
			fprintf( gmres_stats.diagnostic, "P%d : Begining inner GMRES iterations\n", this_dom );
			fflush( gmres_stats.diagnostic );
		}
		else
		{
			gmres_stats.diagnostic = NULL;
		}
		
		// initialise the solution vector
		vec_dist_init_vec( y, x );		
		
		// solve with GMRES : y = inv(S)*x
		MPI_Barrier( P->This.comm );
		gmres( &P->S, x, y, &gmres_stats, &P->MS, 0 );
		//precon_apply( &P->MS, x, y );
		
		// free up memory associated with the GMRES run
		// have to get craftier with all of this when the time comes
		free( gmres_stats.residuals );
		free( gmres_stats.errors );
		free( gmres_stats.K );
		free( gmres_stats.j );
		
		if( gmres_stats.diagnostic && gmres_stats.diagnostic!=stdout )
			fclose( gmres_stats.diagnostic );
		return;
	}
	
	
	/*
	 *
	 *			NOTE TO MYSELF
	 *
	 *			DO I NEED SO MUCH FANCY COMMUNICATION? 
	 *
	 *			How about just doing a vector gather all to each CPU? Then each CPU
	 *			can construct its own vectors using the global permutation vectors.
	 *
	 *			Same goes for the backwards part, replacing all of the complex
	 *			asyncronous communication with two simple gathers. Certainly much friendlier
	 *			on the communication and my debugging!
	 *
	 *			This would work!
	 */
	
	/*
	 *		Apply the forward preconditioning step
	 */ 
	
	if( !P->initial_decomp )
	{
		request = malloc( 2*n_dom*sizeof(MPI_Request) );
		status = malloc( 2*n_dom*sizeof(MPI_Status) );
		
		// permute the x vector into dbl_temp for forward communication
		tmp_double = (double *)malloc( x->n * sizeof(double) );
		permute( x->dat, tmp_double, P->forward.indx, x->n, 1 );	
		
		// initialise memory
		vec_dist_init( &xl, &P->This, P->n_local, P->S.mtx.block, P->domains.vtxdist );
		
		// send information needed by other Pid to form their xl
		for( k=0; k<n_dom; k++ )
			if( P->forward.counts[k] )																							  
				MPI_Isend( tmp_double + P->forward.starts[k], P->forward.counts[k], MPI_DOUBLE, k, TAG_DAT, P->This.comm, request + k );
		
		// make sure that we wait for all the data to be sent before receiving it
		MPI_Barrier( P->This.comm );
		
		// receive information from other Pid to form my local part of xl
		for( k=0; k<n_dom; k++ )
			if( P->backward.counts[k] )
			{
				MPI_Irecv( xl.dat + P->backward.starts[k], P->backward.counts[k], MPI_DOUBLE, k, TAG_DAT, P->This.comm, request + (k+n_dom) );
			}
				
				// wait for communication to finish
				MPI_Waitall(2*n_dom, request, status);
		
		// reorder the xl values to the local node ordering
		tmp_double = realloc( tmp_double, P->n_local*sizeof(double) );
		permute( xl.dat, tmp_double, P->backward.indx, xl.n, 0 ); 
		
		// now redistribute the vector on the local communicator for local preconditioning
		// at the moment assume the communicator is of size 1
		if( P->This.sub_comm->n_proc!=1 )
		{
			fprintf( stderr, "ERROR : precon_schur_apply() : local communicator must be of size 1 at the moment\n" );
			MPI_Finalize();
			exit(1);
		}
		
		vtxdist_in = (int *)malloc( 2*sizeof(int) );
		vtxdist_in[0] = 0;
		vtxdist_in[1] = P->n_in;
		vec_dist_init( &xL, P->This.sub_comm, P->n_in,  P->S.mtx.block, vtxdist_in );
		vec_dist_init_vec( &PxL, &xL );
		memcpy( xL.dat, tmp_double, P->n_in*sizeof(double) );
	}
	else
	{
		vtxdist_in = (int *)malloc( 2*sizeof(int) );
		vtxdist_in[0] = 0;
		vtxdist_in[1] = P->n_in;
		vec_dist_init( &xL, P->This.sub_comm, P->n_in,  P->S.mtx.block, vtxdist_in );
		vec_dist_init_vec( &PxL, &xL );
		if( block )
			memcpy( xL.dat, x->dat, (P->n_in BLOCK_V_SHIFT)*sizeof(double) );
		else
			memcpy( xL.dat, x->dat, P->n_in*sizeof(double) );
	}
	
	// apply the local preconditioner to the vector entries corresponding the the local nodes
	precon_apply( &P->MB, &xL, &PxL );	
	
	// allocate memory for the vector to pass to the next level
	vtxdist_bnd = (int *)malloc( (n_dom+1)*sizeof(int) );
	vtxdist_bnd[0] = 0;
	for( i=0; i<n_dom; i++ )
		vtxdist_bnd[i+1] = vtxdist_bnd[i] + P->domains.n_bnd[i];
	vec_dist_init( &v, &P->This, P->n_bnd, P->S.mtx.block, vtxdist_bnd );
	
	// v = xb
	if( P->initial_decomp )
		tmp_double = x->dat;
	
	if( block )
		memcpy( v.dat, tmp_double + (P->n_in BLOCK_V_SHIFT), (P->n_bnd BLOCK_V_SHIFT)*sizeof(double) );	
	else
		memcpy( v.dat, tmp_double + P->n_in, P->n_bnd*sizeof(double) );

    // v = v - F*PxL
	mtx_CRS_gemv( &P->A.F, PxL.dat, v.dat, -1, 1, 'n' );
	
	// free memory used in all of this
	free( vtxdist_in );
	free( vtxdist_bnd );
	if( !P->initial_decomp )
		free( tmp_double );
	
	/*
	 *		Pass the forward preconditioned boundary data onto the next level of the Schur preconditioner
	 */
	precon_schur_apply( (Tprecon_Schur_ptr)P->MS.preconditioner, &v, &yB );
	
	/*
	 *		Perform the backward preconditioning step 
	 */

	// apply backward preconditioner
	
	// xL = xL - E*yB
	mtx_CRS_gemv( &P->A.E, yB.dat, xL.dat, -1, 1, 'n' );
	
	// yL = B^(-1)*xL
	vec_dist_init_vec( &yL, &xL );
	precon_apply( &P->MB, &xL, &yL );
	
	if( !P->initial_decomp )
	{	
		tmp_double = (double *)malloc( P->n_local*sizeof(double) );
		dat = (double*)malloc( P->n_local*sizeof(double) );
		
		// copy backward preconditioned y into a single vector
		memcpy( tmp_double,           yL.dat, P->n_in*sizeof(double) );
		memcpy( tmp_double + P->n_in, yB.dat, P->n_bnd*sizeof(double) );
		
		// reorder y values from local to global node order
		permute( tmp_double, dat, P->backward.indx, xl.n, 1 );
		
		// communicate information to other Pid
		for( k=0; k<n_dom; k++ )
			if( P->backward.counts[k] )
				MPI_Isend( dat + P->backward.starts[k], P->backward.counts[k], MPI_DOUBLE, k, TAG_DAT, P->This.comm, request + k );
		
		// receive information from other Pid to form my local part of xl
		tmp_double = (double *)realloc( tmp_double, x->n*sizeof(double) );
		for( k=0; k<n_dom; k++ )
			if( P->backward.counts[k] )
				MPI_Irecv( tmp_double + P->forward.starts[k], P->forward.counts[k], MPI_DOUBLE, k, TAG_DAT, P->This.comm, request + (k+n_dom) );
		
		// wait for communication to finish
		MPI_Waitall(2*n_dom, request, status);
		
		/*
		 *		another permutation needed here
		 */
		vec_dist_init_vec( y, x );
		permute( tmp_double, y->dat, P->forward.indx, y->n, 0 );
		
		// free memory only used in the undecomposed form
		free( dat );
		free( tmp_double );
		free( status );
		free( request );
	}
	else
	{
		vec_dist_init_vec( y, x );
		
		if( block )
		{
			memcpy( y->dat,           yL.dat, (P->n_in  BLOCK_V_SHIFT)*sizeof(double) );
			memcpy( y->dat + (P->n_in BLOCK_V_SHIFT), yB.dat, (P->n_bnd BLOCK_V_SHIFT)*sizeof(double) );
		}
		else
		{
			memcpy( y->dat,           yL.dat, P->n_in*sizeof(double) );
			memcpy( y->dat + P->n_in, yB.dat, P->n_bnd*sizeof(double) );
		}
	}

	
	/*
	 *		free up all memory
	 */
	vec_dist_free( &xl );
	vec_dist_free( &yB );
	vec_dist_free( &xL );
	vec_dist_free( &yL );
	vec_dist_free( &PxL );
}

void GMRES_setup_schur( Tgmres_ptr g )
{
	g->dim_krylov = 25;
	g->max_restarts = 100;
	g->tol = 1e-10;
	g->restarts = 0;
	g->nrmcols = 0;
	g->residuals = (double *)malloc( (g->max_restarts+1)*sizeof(double) );
	g->errors = (double *)malloc( (g->max_restarts+1)*sizeof(double) );
	g->j = (int *)malloc( (g->max_restarts+1)*sizeof(int) );
	g->K = (int *)malloc( (g->max_restarts+1)*sizeof(int) );
	g->k = 0;
	g->time_precon = 0;
	g->time_gmres = 0;
	g->diagnostic = NULL;
	g->reorthogonalise = 0;
}
void precon_schur_free( Tprecon_Schur_ptr P )
{
	P = (P+1)-1;
#ifdef DEBUG
	//if( P->init )
	//	printf( "precon_schur_free() : not yet written\n" );
#endif
}

void mtx_CRS_dist_split_order( Tmtx_CRS_dist_ptr A, Tprecon_Schur_ptr P, Tmtx_CRS_dist_ptr Aschur )
{
	FILE *fid;
	char fname[50];
	int i, j, k, pos, this_dom, tdom, sdom, r_pos, n_nodes, jj;
	int tag, index_pos, recv_nodes, is_boundary, nnz, success, success_global;
	int n_in, n_bnd, n_node, nodes_max, nnz_max, n_dom;
	int *is_neigh, *back_counts, *index_pack, *tag_pack, *nnz_pack, *tag_pack_recv, *nnz_pack_recv, *index_pack_recv;
	int *p, *int_temp;
	int block, n_neigh, connect_dom, max_node_nnz=0;
	double *nz_pack, *dbl_temp, *nz_pack_recv;
	Tnode_ptr  node=NULL;
	Tnode_list in, bnd, nodes;
	TMPI_dat_ptr This;
	MPI_Request *request;
	MPI_Status *status;
	int nodes_myself, nodes_others, *first_error, *last_error, *l_recv;
	Tmtx_CRS_ptr As;
	
	/*
	 *		Setup the diagnostic file for This thread
	 *		
	 *		This file contains diagnostic output describing the communication
	 *		between processes/domains and the domain decompt method
	 */
	
	As = (Tmtx_CRS_ptr)malloc( sizeof(Tmtx_CRS) );
	A->init = 0;
	This = &A->This;
	sprintf( fname, "P%d_split.txt", This->this_proc );
	fid = fopen( fname, "w" );
	
	/*
	 *		Setup initial data stuff
	 */
	
	// setup variables
	this_dom    = This->this_proc;
	block       = A->mtx.block;
	n_dom = This->n_proc;
	back_counts = (int*)malloc( sizeof(int)*n_dom );
	
	// initialise data types
	node_list_init( &in,    block );
	node_list_init( &bnd,   block );
	node_list_init( &nodes, block );
	
	// domain decomp using ParMETIS
	mtx_CRS_dist_domdec( A, &P->forward, P->part_g );

	/* 
	 *		Send and recieve nodes
	 *
	 *		The processors exchange nodes with one another so that at completion of This
	 *		loop each processor has only the nodes corresponding to its domain.
	 *		On each iteration of the loop P(i) will give nodes to P(i+k) and recieve
	 *		nodes from P(i-k), allowing all processes to continue communicating 
	 *		constantly.
	 *
	 *		The nodes are stored in node linked lists, not in CRS format, This allows
	 *		easy sorting and manipulation of the nodes.
	 */
	
	request = malloc( 20*sizeof(MPI_Request) );
	status  = malloc( 20*sizeof(MPI_Status) );
	for( i=0; i<20; i++ )
		status[i].MPI_ERROR = MPI_SUCCESS;
	nodes_max = 0;
	for( k=0; k<n_dom; k++ )
		if( nodes_max<P->forward.counts[k] )
			nodes_max = P->forward.counts[k];
	tag_pack		= (int*)malloc( nodes_max*sizeof(int) );
	nnz_pack		= (int*)malloc( nodes_max*sizeof(int) );
	tag_pack_recv   = (int*)malloc( nodes_max*sizeof(int) );
	nnz_pack_recv   = (int*)malloc( nodes_max*sizeof(int) );
	
	nnz_max    = A->mtx.nnz;
	if( A->mtx.block )
	{
		nz_pack    = (double*)malloc( (nnz_max*sizeof(double))BLOCK_M_SHIFT );
		nz_pack_recv    = (double*)malloc( (nnz_max*sizeof(double))BLOCK_M_SHIFT );
	}
	else
	{
		nz_pack    = (double*)malloc( nnz_max*sizeof(double) );
		nz_pack_recv    = (double*)malloc( nnz_max*sizeof(double) );
	}
	index_pack = (int*)malloc( nnz_max*sizeof(int) );
	index_pack_recv = (int*)malloc( nnz_max*sizeof(int) );

	
	// debug stuff
	nodes_myself=nodes_others=0;
	first_error = (int*)malloc( sizeof(int)*n_dom );
	last_error = (int*)malloc( sizeof(int)*n_dom );
	l_recv = (int*)malloc( sizeof(int)*n_dom );
		
	for( k=0; k<n_dom; k++ )
	{
		r_pos = 0;
		
		// determine the target and source domains
		tdom = k +  this_dom;
		if( tdom>=n_dom )	
			tdom -= n_dom;
		
		sdom = this_dom-k;
		if( sdom<0 )		
			sdom += n_dom;
		
		/*
		 *		Pack the data for sending
		 */
		
		// the number of nodes to send to sdom
		n_nodes = P->forward.counts[tdom];	
		nnz = 0;
		
		
		for( pos=0, i=P->forward.starts[tdom]; i<P->forward.starts[tdom+1]; pos++, i++ )
		{
			index_pos = A->mtx.rindx[P->forward.indx[i]];
			
			tag_pack[pos]   = A->vtxdist[this_dom] + P->forward.indx[i];
			nnz_pack[pos]   = A->mtx.rindx[P->forward.indx[i]+1] - index_pos;
			memcpy( index_pack + nnz, A->mtx.cindx + index_pos, nnz_pack[pos]*sizeof(int) );
			if( block )
				memcpy( nz_pack + (nnz BLOCK_M_SHIFT),    A->mtx.nz + (index_pos BLOCK_M_SHIFT),    (nnz_pack[pos]*sizeof(double))BLOCK_M_SHIFT );
			else
				memcpy( nz_pack + (nnz),    A->mtx.nz + (index_pos),    nnz_pack[pos]*sizeof(double) );

			nnz+=nnz_pack[pos];
		}
		
		
		fprintf( fid, "P%d : sending to P%d : %d nodes with %d indices\n", this_dom, tdom, n_nodes, nnz );		
		/*
		 *		Send information to target Pid
		 */
		
		// send the number of nodes being passed
		MPI_Isend( &n_nodes, 1,      MPI_INT, tdom, TAG_NODES, This->comm, request+r_pos );
		r_pos++;
		
		if( n_nodes )
		{
			// tags
			MPI_Isend( tag_pack, n_nodes, MPI_INT, tdom, TAG_TAG, This->comm, request+r_pos );
			r_pos++;
			
			// nnz
			MPI_Isend( nnz_pack,   n_nodes, MPI_INT, tdom, TAG_NNZ, This->comm, request+r_pos );
			r_pos++;
			
			// indices
			MPI_Isend( index_pack,  nnz,     MPI_INT, tdom, TAG_INDEX, This->comm, request+r_pos );
			r_pos++;
			
			// data
			if( A->mtx.block )
				MPI_Isend( nz_pack, nnz BLOCK_M_SHIFT,     MPI_DOUBLE, tdom, TAG_NZ, This->comm, request+r_pos );
			else
				MPI_Isend( nz_pack, nnz,                   MPI_DOUBLE, tdom, TAG_NZ, This->comm, request+r_pos );
			r_pos++;
		}
		
		
		/*
		 *		Receive data from the source Pid
		 */
		
		//  wait for the source to have sent its data to you, if you jump the gun here
		//	things go pear shaped pretty quickly
		MPI_Barrier( This->comm );
		
		// number of nodes
		MPI_Irecv( &recv_nodes, 1, MPI_INT, sdom, TAG_NODES , This->comm, request+r_pos );
		r_pos++;
		
		// check that we aren't going to run out of memory
		if( recv_nodes>nodes_max )
		{
			nodes_max = recv_nodes;
			tag_pack_recv = (int*)realloc( tag_pack_recv, nodes_max*sizeof(int) );
			if( block )
				nnz_pack_recv = (int*)realloc( nnz_pack_recv, (nodes_max*sizeof(int)) BLOCK_M_SHIFT );
			else
				nnz_pack_recv = (int*)realloc( nnz_pack_recv, nodes_max*sizeof(int) );
		}
		
		back_counts[ sdom ] = recv_nodes;
		
		if( recv_nodes )
		{
			// tags
			MPI_Irecv( tag_pack_recv, recv_nodes, MPI_INT, sdom, TAG_TAG, This->comm, request+r_pos );
			r_pos++;
						
			// nnz
			MPI_Irecv( nnz_pack_recv, recv_nodes, MPI_INT, sdom, TAG_NNZ, This->comm, request+r_pos );
			r_pos++;
						
			// calculate the number of indices being received
			nnz = 0;
			for( i=0; i<recv_nodes; i++ )
				nnz += nnz_pack_recv[i];
			
			// check that we have enough memory in the buffer
			if( nnz>nnz_max )
			{
				printf( "P%d : had to allocate more memory for the MPI receive buffers, receiving from P%d\n\n", this_dom, sdom );
				
				nnz_max = nnz;
				index_pack_recv =    (int*)realloc( index_pack_recv, nnz_max*sizeof(int) );
				if( block )
					nz_pack_recv    = (double*)realloc( nz_pack_recv,    (nnz_max*sizeof(double))BLOCK_M_SHIFT );
				else
					nz_pack_recv    = (double*)realloc( nz_pack_recv,    nnz_max*sizeof(double) );
			}
						
			// indices
			MPI_Irecv( index_pack_recv,  nnz, MPI_INT, sdom, TAG_INDEX, This->comm, request+r_pos );
			r_pos++;
			
			// nz values
			if( block )
				MPI_Irecv( nz_pack_recv,  nnz BLOCK_M_SHIFT,   MPI_DOUBLE, sdom, TAG_NZ, This->comm, request+r_pos ); 
			else
				MPI_Irecv( nz_pack_recv,  nnz,   MPI_DOUBLE, sdom, TAG_NZ, This->comm, request+r_pos ); 
			r_pos++;
		}
				
		// wait for all communication to finish
		for( i=0; i<r_pos; i++ )
		{
			MPI_Wait( request+i, status+i );
			if( status[i].MPI_ERROR!=MPI_SUCCESS)
				printf( "P%d -- WARNING : status for a message %d was not MPI_SUCCESS\n", this_dom, i );
		}
		
		/*
		 *		Unpack data into node list
		 */
		jj = 0;
		first_error[k] = last_error[k] = -1;
		l_recv[k] = recv_nodes; 
		for( i=0, nnz=0; i<recv_nodes; i++ )
		{
			int ii;
			
			// allocate memory for the new node
			if( !node_list_add( &nodes, nnz_pack_recv[i], tag_pack_recv[i] ) )
				fprintf( stderr, "\tP%d : ERROR, unable to add node with tag %d\n", this_dom, tag );
			
			// copy over the indices and nz values
			memcpy( nodes.opt->indx, index_pack_recv + nnz, sizeof(int)*nnz_pack_recv[i] );
			if( block )
				memcpy( nodes.opt->dat, nz_pack_recv + (nnz BLOCK_M_SHIFT), (sizeof(double)*nnz_pack_recv[i]) BLOCK_M_SHIFT );
			else
				memcpy( nodes.opt->dat, nz_pack_recv + nnz, sizeof(double)*nnz_pack_recv[i] );

			nnz += nnz_pack_recv[i];
			
			ii=0;
			while( nodes.opt->indx[ii] != nodes.opt->tag && ii<nodes.opt->n )
				ii++;
			if( nodes.opt->dat[ii BLOCK_M_SHIFT]!=5 )
			{
				if( first_error[k]==-1 )
					first_error[k] = i;
				last_error[k] = i;
				if( this_dom==sdom )
					nodes_myself++;
				else
					nodes_others++;
				jj++;
			}
		}		
	}
	
	if( jj )
	{
		printf( "P%d : %d errors\n", this_dom, jj );
		for( i=0; i<n_dom; i++ )
			printf( "\tPid %d\tk = %d\t[%d %d] in [0 %d]\n", this_dom, i, first_error[i], last_error[i], l_recv[i] );
	}
	
	node = nodes.start;
	while( node!=NULL )
	{
		if( node->n>max_node_nnz )
			max_node_nnz = node->n;
		node = node->next;
	}
	free( index_pack );
	free( nnz_pack );
	free( tag_pack );
	free( nz_pack );
	free( index_pack_recv );
	free( nnz_pack_recv );
	free( tag_pack_recv );
	free( nz_pack_recv );
	free( request );
	free( status  );
	
	// everyone checks that they have recieved valid node lists
	success = node_list_verify( NULL, &nodes );
	MPI_Allreduce( &success, &success_global, 1, MPI_INT, MPI_LAND, This->comm );
	if( !success_global )
	{
		fprintf( fid, "ERROR : one of the processes had an invalid node list\n" );
		return;
	}
	
	/*
	 *		Determine the internal and boundary nodes for This domain.
	 *
	 *		Internal and boundary nodes are sorted into the lists in and out
	 *		respectively. The element is_neigh[i] records how many edges This
	 *		domain shairs with domain i. Remember, the number of edges is NOT
	 *		the number of nodes connected to. Think of is_neigh[i] as the nnz
	 *		in the Ei for This domain.
	 */
	is_neigh = (int *)calloc( n_dom, sizeof(int) );
	nnz = 0;
	p = P->part_g;
	pos = 1;
	while( (node = node_list_pop( &nodes )) != NULL )
	{
		is_boundary = 0;
		
		// does the node contain references to external nodes?
		for( i=0; i<node->n; i++ )
		{
			// is This an external reference?
			connect_dom = p[node->indx[i]];
			if( connect_dom!=this_dom )
			{
				// it is so add the domain to our list of neighbours
				is_neigh[connect_dom]++;
				
				// flag This node as being on the boundary
				is_boundary = 1;
			}
		}
		
		// store the node in the relevant list
		if( is_boundary )
			node_list_push_start( &bnd, node );
		else
			node_list_push_start( &in, node );
		nnz += node->n;
		pos ++;
		fprintf( fid, "\n" );
	}
		
	// determine the number of nodes that are in/out
	P->n_in  = n_in   = in.n;
	P->n_bnd = n_bnd  = bnd.n;

	P->n_local = n_in + n_bnd;
	n_node  = n_in + n_bnd;
	is_neigh[this_dom] = n_in;
	
	// determine how many domains we are connected to
	n_neigh = 0;
	for( k=0; k<n_dom; k++ )
		if( is_neigh[this_dom] )
			n_neigh++;
	P->n_neigh = n_neigh;
	
	// print out the interior and boundary node data
	fprintf( fid, "\nFinished categorising nodes : I have %d interior and %d boundary nodes\n", in.n, bnd.n );
	print_nodes( fid, "interior", &in );
	print_nodes( fid, "boundary", &bnd );
	
	/*
	 *		Processes send one-another information on who they are connected to
	 *		
	 *		This information is stored in map, where map[i*ndom:(i+1)*ndom-1] contains the is_neigh
	 *		array for domain i.
	 *
	 *		The processors let everyone else know how many nodes they now have, allowing each
	 *		process to build a new vtxdist for the new domain decomposition.
	 */
	domain_init( &P->domains, This, n_node );
	
	// MPI command to distribute information
	MPI_Allgather( is_neigh, n_dom, MPI_INT, P->domains.map,     n_dom, MPI_INT, This->comm );
	
	// output the map to the output file
	// PRINTGLOBALCONNECTIONS;
	
	// determine vtxdist for This distribution
	int_temp = (int *)malloc( sizeof(int)*n_dom );
	MPI_Allgather( &n_node,  1,     MPI_INT, int_temp, 1,     MPI_INT, This->comm );
	
	P->domains.vtxdist[0] = 0;
	for( i=1; i<=n_dom; i++ )
	{
		P->domains.vtxdist[i] = P->domains.vtxdist[i-1]+int_temp[i-1];
	}
	free( int_temp );
	
	// update all processors on the numbner of interior and boundary nodes
	// in each domain
	MPI_Allgather( &n_in,  1, MPI_INT, P->domains.n_in,  1, MPI_INT, This->comm );
	MPI_Allgather( &n_bnd, 1, MPI_INT, P->domains.n_bnd, 1, MPI_INT, This->comm );
	
	/*
	 *  Find the new node order of This matrix, and store in the backwards distribution structure
	 *
	 *  among other things This information is used when passing vectors between processors on the bsub
	 *  phase of preconditioning
	 */
	
	// intialise the data structure
	distribution_init( &P->backward, n_dom, n_node, n_neigh );
	for( pos=0, k=0; pos<n_neigh; k++ )
	{
		if( is_neigh[k] )
			P->backward.neighbours[pos++] = k;
	}
	
	// copy the tags and other data into the distribution arrays
	for( i=0; i<n_node; i++ )
		P->backward.indx[i] = i;
	node = in.start;
	for( i=0; i<n_in; i++ )
	{
		P->backward.part[i] = P->backward.ppart[i] = node->tag;
		node = node->next;
	}
	node = bnd.start;
	for( ; i<n_node; i++ )
	{
		P->backward.part[i] = P->backward.ppart[i] = node->tag;
		node = node->next;
	}
	
	// sort the part so that indx holds the permution vector
	heapsort_int_index( n_node, P->backward.indx, P->backward.part);
	
	// how many nodes were received from each domain has already been calculated
	// so just swap it into the distribution array
	free( P->backward.counts );
	P->backward.counts    = back_counts;
	P->backward.starts[0] = 0;
	for( i=1; i<=n_dom; i++ )
	{
		P->backward.starts[i] = P->backward.starts[i-1] + P->backward.counts[i-1];
	}
	
	/* 
	 *		calculate and store the global permutation vector
	 */
	
	// allocate memory for temporary working arrays
	int_temp = (int *)malloc( sizeof(int)*n_dom );
	p        = (int *)malloc( sizeof(int)*A->nrows );
	
	// determine how many nodes are stored on each processor
	for( i=0; i<n_dom; i++ )
		int_temp[i] = P->domains.vtxdist[i+1] - P->domains.vtxdist[i];
	
	// gather each domain's local ordering for the original node tags
 	MPI_Allgatherv( P->backward.ppart, int_temp[this_dom], MPI_INT, p, int_temp, P->domains.vtxdist, MPI_INT, This->comm );
	
	// convert into global permutation
	for( i=0; i<A->nrows; i++ )
		P->q[p[i]] = i;
	
	// output p
	if( !this_dom )
	{
		FILE *fidp;
		
		if( (fidp = fopen( "p.txt", "w" ))==NULL )
			fprintf( stderr, "There is strife with the file\n" );
		
		for( i=0; i<A->nrows; i++ )
			fprintf( fidp, "%d\t%d\n", p[i], P->q[i] );
		
		fclose( fidp );
	}
	MPI_Barrier( This->comm );
	
	// free temporary work arrays
	free( int_temp );
	free( p );
	
	/*
	 *		make the nodes' cindx values in the new node order
	 */
	
	int_temp = (int*)malloc( sizeof(int)*max_node_nnz );
	nnz = 0;
	
	// internal nodes
	node = in.start;
	while( node )
	{
		nnz += node->n;
		
		// sort the indices, keeping a permutation array
		for( i=0; i<node->n; i++ )
		{
			node->indx[i] = P->q[node->indx[i]];
			int_temp[i] = i;
		}
		heapsort_int_index( node->n, int_temp, node->indx);

		// sort the nonzero values according to the permuation array
		if( block )
		{
			dbl_temp = (double *)malloc( sizeof(double)*(node->n BLOCK_M_SHIFT) );
			permuteB( node->dat, dbl_temp, int_temp, node->n, 1 );
		}
		else
		{
			dbl_temp = (double *)malloc( sizeof(double)*node->n );
			permute( node->dat, dbl_temp, int_temp, node->n, 1 );
		}
		free( node->dat );
		node->dat = dbl_temp;
				
		node = node->next;
	}

	// boundary nodes
	node = bnd.start;
	while( node )
	{
		nnz += node->n;
		
		// sort the indices, keeping a permutation array
		for( i=0; i<node->n; i++ )
		{
			node->indx[i] = P->q[node->indx[i]];
			int_temp[i] = i;
		}
		heapsort_int_index( node->n, int_temp, node->indx);
		
		// sort the nonzero values according to the permuation array
		if( block )
		{
			dbl_temp = (double *)malloc( sizeof(double)*(node->n BLOCK_M_SHIFT) );
			permuteB( node->dat, dbl_temp, int_temp, node->n, 1 );
		}
		else
		{
			dbl_temp = (double *)malloc( sizeof(double)*node->n );
			permute( node->dat, dbl_temp, int_temp, node->n, 1 );
		}
		
		free( node->dat );
		node->dat = dbl_temp;
		
		node = node->next;
	}

	free( int_temp );
	
	// allocate space for the new matrix
	mtx_CRS_init( As, n_in+n_bnd, A->ncols, nnz, block );
	
	/*
	 *		Store the nodes in the new matrix
	 */
	
	j=0;
	pos = 0;
	As->rindx[0] = 0;
	
	// internal nodes
	node = in.start;
	while( node )
	{
		As->rindx[j+1] = As->rindx[j]+node->n;
		memcpy( As->cindx+pos, node->indx, node->n*sizeof(int) );
		if( !block )
			memcpy( As->nz+pos, node->dat, node->n*sizeof(double) );
		else
			memcpy( As->nz+(pos BLOCK_M_SHIFT), node->dat, node->n*sizeof(double) BLOCK_M_SHIFT );	

		pos+=node->n;
		j++;
		node = node->next;
	}
	// external nodes
	node = bnd.start;
	while( node )
	{
		As->rindx[j+1] = As->rindx[j]+node->n;
		memcpy( As->cindx+pos, node->indx, node->n*sizeof(int) );
		if( !block )
			memcpy( As->nz+pos, node->dat, node->n*sizeof(double) );
		else
			memcpy( As->nz+(pos BLOCK_M_SHIFT), node->dat, node->n*sizeof(double) BLOCK_M_SHIFT );
		
		pos+=node->n;
		j++;
		node = node->next;
	}	
	
	/*
	 *		Build the global matrix
	 */
	mtx_CRS_dist_init( Aschur, As->nrows, As->ncols, 1, As->block, This );
	mtx_CRS_copy( As, &Aschur->mtx );
}

void find_i( int *v, int n )
{
	int i;
	
	for( i=0; i<n; i++ )
		if( v[i] )
			printf( "\t%d", i );
	printf( "\n" );
}

void print_sparsity( double *v, int n, char *tag )
{
	int *sparsity;
	int i, pos;
	double *s;
	
	sparsity = (int*)malloc( sizeof(int)*n );
	
	for( i=0; i<n; i++ )
		sparsity[i] = 0;
	for( i=0; i<BLOCK_SIZE; i++ )
	{
		s = v + i*n*BLOCK_SIZE;
		for( pos=0; pos<n; pos++, s+=BLOCK_SIZE )
			if( BLOCK_V_ISNZ( s ) )
				sparsity[pos] = 1;
	}
	printf( "sparsity for %s\n", tag );
	find_i( sparsity, n );
	
	free( sparsity );
}

void print_sparsity_nz( double *v, int n, char *tag )
{
	int *sparsity;
	int i, pos;
	double *s;
	
	sparsity = (int*)malloc( sizeof(int)*n );
	
	for( i=0; i<n; i++ )
		sparsity[i] = 0;
	for( i=0; i<BLOCK_SIZE; i++ )
	{
		s = v + i*n*BLOCK_SIZE;
		for( pos=0; pos<n; pos++, s+=BLOCK_SIZE )
			if( BLOCK_V_ISNZ( s ) )
			{
				sparsity[pos] = 1;
			}
	}
	printf( "sparsity for %s\n", tag );
	find_v( sparsity, v, n );
	printf( "\n" );
	free( sparsity );
}

void find_v( int *v, double *u, int n )
{
	int i;
	double *vv;
	
	for( i=0; i<n; i++ )
	{
		if( v[i] )
		{
			vv = u + i*BLOCK_SIZE;
			printf( "%d\t%15g\t%15g\n\t%15g\t%15g\n", i, vv[0], vv[n*BLOCK_SIZE], vv[1], vv[n*BLOCK_SIZE+1]  );
		}
	}

}
