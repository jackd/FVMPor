#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>

#include "ben_mpi.h"
#include "linalg_mpi.h"
#include "linalg_dense_mpi.h"
#include "parmetis.h"
#include "linalg.h"
#include "indices.h"
#include "fileio.h"
#include "benlib.h"
#include "precon.h"

int *ppp;

void mesh_decomp( Tmtx_CRS_dist_ptr A, Tprecon_schur_ptr P, Tnode_list_ptr *_nin, Tnode_list_ptr *_nbnd );

void mesh_split( Tmesh_ptr mesh, TMPI_dat *This, int root )
{
	int i, j, thisroot=0, first, last, this_dom, n_dom, n, n_in, n_bnd, n_ext, n_els, k, pos, posn, found;
	int *els_check, n_els_global, **node_els, *node_num_els, *els_list, **dom_els;
	int *ext_list, *ext_check;
	int *dom_n_els;
	double *node_x, *node_y, *node_z;
	Tmtx_CRS_dist A;
	Tprecon_schur P;
	Tnode_ptr node;
	Tnode_list_ptr nodes_in, nodes_bnd;
	Telement_ptr my_elements, el;
	
	// find out processor stuff
	this_dom = This->this_proc;
	n_dom = This->n_proc;	
	if( root==This->this_proc )
		thisroot = 1;
	
	// split the pattern matrix over the Pid
	if( thisroot )
		mtx_CRS_distribute( &mesh->A, &A, This, root );
	else
		mtx_CRS_distribute( NULL, &A, This, root );
	
	// output the jacobian patttern for This Pid's local nodes
	if( thisroot )
		printf( "\toutputting pattern matrices\n" );
	//sprintf( fname, "Jpattern_%d.mtx", This->this_proc );
	//mtx_CRS_output( &A.mtx, fname );
	
	MPI_Barrier( This->comm );
	
	/*
	 *		perform a domain decomposition
	 */
	if( thisroot )
		printf( "\tcalculating domain decomp and redistributing node data\n" );
	precon_schur_init( &P, &A  );
	mesh_decomp( &A, &P, &nodes_in, &nodes_bnd );

	n_in = nodes_in->n;
	n_bnd = nodes_bnd->n;
	
	if( (node = nodes_bnd->start)==NULL )
		printf("P%d : 0 NULL bnd\n",this_dom);
	
	/*
	 *		we know our interior and boundary nodes, now find our exterior nodes
	 */

	ext_list = (int*)malloc( n_bnd*sizeof(int)*5 );
	ext_check = (int*)calloc( A.nrows, sizeof(int) );
	first = A.vtxdist[this_dom];
	last  = A.vtxdist[this_dom+1]-1;
	node = nodes_bnd->start;
	
	if( thisroot )
		printf( "\tdeterming exterior node connections for local domains\n" );
	
	if( (node = nodes_bnd->start)==NULL )
		printf("P%d : 0.1 NULL bnd\n", this_dom);
	
	// loop through the boundary nodes and check for exterior nodes they connect to
	if( thisroot )
		printf( "node = %d\n", node );
	while( node!=NULL )
	{
		for( i=0; i<node->n; i++ )
		{
			n = node->indx[i];
			if( n<first || n>last )
			{
				if( !ext_check[n] )
				{
					ext_list[n_ext++] = n;
					ext_check[n] = 1;
				}
			}
#ifdef DEBUG
			if( n_ext>n_bnd*5 )
			{
				printf( "ERROR : mesh_split() : need to allocate more memory for ext_list, allocated %d and need %d\n", n_ext, n_bnd*5 );
				exit(1);
			}
#endif
		}
		
		node = node->next;
	}
	
	if( (node = nodes_bnd->start)==NULL )
		printf("P%d : 1 NULL bnd\n",this_dom);
	
	// sort the list of exterior nodes

	heapsort_int( n_ext, ext_list );
	free( ext_check );
	ext_list = realloc( ext_list, n_ext*sizeof(int)  );
	
	/*
	 *		send everyone all of the node position and element data
	 */
	
	if( thisroot )
		printf( "\troot communicating node position and element data with other Pid data \n" );
	
	if( thisroot )
	{
		node_x = mesh->node_x;
		node_y = mesh->node_y;
		node_z = mesh->node_z;
		node_els = mesh->node_elements;
		node_num_els = mesh->node_num_elements;
	}
	else
	{
		node_x = (double*)malloc( A.nrows*sizeof(double) );
		node_y = (double*)malloc( A.nrows*sizeof(double) );
		node_z = (double*)malloc( A.nrows*sizeof(double) ); 
		node_els = (int**)malloc( A.nrows*sizeof(int*) );
		node_num_els = (int*)malloc( A.nrows*sizeof(int) );
	}
	if( (node = nodes_bnd->start)==NULL )
		printf("P%d : NULL PROBLEM HAS MOVED FORWARD : BAAAAD\n", this_dom );
	MPI_Bcast( node_x, A.nrows, MPI_DOUBLE, root, This->comm );
	
	if(  (node = nodes_bnd->start)==NULL )
		printf("P%d : NULL PROBLEM EXISTS\n", this_dom);
	MPI_Bcast( node_y, A.nrows, MPI_DOUBLE, root, This->comm );
	
	MPI_Bcast( node_z, A.nrows, MPI_DOUBLE, root, This->comm );	
	MPI_Bcast( node_num_els, A.nrows, MPI_INT, root, This->comm );
	
	if( !thisroot )
	{
		for( i=0; i<A.nrows; i++ )
			node_els[i] = (int*)malloc( node_num_els[i]*sizeof(int) );
	}
	for( i=0; i<A.nrows; i++ )
		MPI_Bcast( *(node_els+i), node_num_els[i], MPI_INT, root, This->comm );
	
	/*
	 *		This is a big job, each process has to determine which elements
	 *		belong to it...
	 */ 
	if( thisroot )
		n_els_global = mesh->n_elements;
	MPI_Bcast( &n_els_global, 1, MPI_INT, root, This->comm );
	
	// This array tells me if I have already checked in This element
	els_check = (int*)calloc( n_els_global, sizeof(int) );
	els_list = (int*)calloc( n_els_global, sizeof(int) );
	n_els = 0;
	
	// check every node's elements to see which it is connected to
	for( i=0; i<n_in+n_bnd; i++ )
	{
		n = P.p[first+i];
		for( j=0; j<node_num_els[n]; j++ )
		{
			if( !els_check[node_els[n][j]] )
			{
				els_check[node_els[n][j]] = 1;
				els_list[n_els++] = node_els[n][j];
			}
		}
	}

	
	// sort the elements in This domain
	heapsort_int( n_els, els_list );
	els_list = (int*)realloc( els_list, n_els*sizeof(int) );
	
	// let the master know how many we each have
	if( thisroot )
		dom_n_els = (int*)malloc( n_dom*sizeof(int) );
	MPI_Gather( &n_els, 1, MPI_INT, dom_n_els, 1, MPI_INT, root, This->comm );
	
	// everyone tell the root exactly which ones we have
	if( thisroot )
	{
		dom_els = (int**)malloc( n_dom*sizeof(int*) );
		for( i=0; i<n_dom; i++ )
			dom_els[i] = (int*)malloc( dom_n_els[i]*sizeof(int) );
	}
	if( thisroot )
	{
		MPI_Status status;
		
		for( i=0; i<n_dom; i++ )
		{
			if( i!=root )
				MPI_Recv( dom_els[i], dom_n_els[i], MPI_INT, i, i, This->comm, &status );
			else
				memcpy( dom_els[i], els_list, n_els*sizeof(int) );
		}
	}
	else
	{
		MPI_Send( els_list, n_els, MPI_INT, root, this_dom, This->comm );
	}
	
	// send out the information to each Pid
	if( thisroot )
	{
		for( i=0; i<n_dom; i++ )
		{
			// sending to myself
			if( i==root )
			{
				my_elements = (Telement_ptr)malloc( n_els*sizeof(Telement) );
				el = my_elements;
				
				for( n=0; n<n_els; n++ )
				{
					memcpy( el, mesh->elements + dom_els[i][n], sizeof(Telement) );
					//el = mesh->elements + dom_els[i][n];
					
					// send element structure
					el ++;
				}
			}
			else
			{
				for( n=0; n<dom_n_els[i]; n++ )
				{
					el = mesh->elements + dom_els[i][n];
					
					// send element structure
					MPI_Send( el, sizeof(Telement), MPI_BYTE, i, n, This->comm );
					
					// send the node and face data
					MPI_Send( el->nodes, el->n_nodes, MPI_INT, i, n, This->comm );
					MPI_Send( el->face_bcs, el->n_faces, MPI_INT, i, n, This->comm );
				}
			}
		}
	}
	// receive data from root Pid
	else		
	{
		MPI_Status status;
		
		my_elements = (Telement_ptr)malloc( n_els*sizeof(Telement) );
		el = my_elements;
		
		for( n=0; n<n_els; n++ )
		{
			// receive element structure
			MPI_Recv( el, sizeof(Telement), MPI_BYTE, root, n, This->comm, &status );
			
			// allocate memory for element arrays
			el->nodes    =  (int*)malloc( el->n_nodes*sizeof(int) );
			el->face_bcs =  (int*)malloc( el->n_faces*sizeof(int) );
			
			// receive the node and face data
			MPI_Recv( el->nodes, el->n_nodes, MPI_INT, root, n, This->comm, &status );
			MPI_Recv( el->face_bcs, el->n_faces, MPI_INT, root, n, This->comm, &status );
			
			el++;
		}
	}
	
#ifdef DEBUG
	
	/*
	 *		verify that each node is in the elements it claims to be
	 */
	
	// check that the elements have been passed in-order
	for( i=0; i<n_els; i++ )
	{
		if( my_elements[i].tag!=els_list[i] )
		{
			printf( "P%d : elements are out of order\n", this_dom );
			break;
		}
	}
	
	els_check = (int*)calloc( n_els, sizeof(int) );
	
	// loop over each node
	for( i=0; i<n_in+n_bnd; i++ )
	{
		// determine the position of the node in the old order
		n = P.p[first+i];
		
		// loop through the elements that the node is meant to be in
		for( k=0; k<node_num_els[n]; k++ )
		{
			pos = node_els[n][k];
			j = binary_search( n_els, els_list, pos );
			if( j<0 )
			{
				printf( "ERROR : P%d : local node %d : requested element %d is not stored locally\n", this_dom, i, pos );
				break;
			}
			els_check[j]=1;
			el = my_elements +j;
			found = 0;
			for( posn=0; posn<el->n_nodes; posn++ )
			{
				if( el->nodes[posn]==n )
				{
					found=1;
					break;
				}
			}
			if( !found )
			{
				printf( "ERROR : node %d does not belong to element %d\n", n, pos );
			}
		}
	}
	
	
	
	pos = 0;
	for( i=0; i<n_els; i++ )
		if( !els_check[i] )
			pos++;
	printf( "P%d has pos %d\n", this_dom, pos );
	if( pos )
		printf( "ERROR : P%d has %d elements that do not contain any of its local nodes\n", this_dom, pos );
	
	
	free( els_check );
	
	/*
	 *		verify that each node is connected to all of the nodes it should be
	 */
	// internal nodes
	if( this_dom==3 && (node = nodes_bnd->start)==NULL )
		printf("NULL bnd\n");
	i = 0;
	while( node!=NULL )
	{
		if( this_dom==3 )
			printf( "\t\tchecking boundary node with tag %d\n", node->tag );
		if( node==NULL )
			printf( "FARQ\n\n\n" );
		
		// find old index
		n = P.p[first+i];
		
		// loop through them elements that the node is connected to
		for( k=0; k<node_num_els[n]; k++ )
		{
			// find the element
			pos = node_els[n][k];
			j = binary_search( n_els, els_list, pos );
			el = my_elements+j;			
			
			// loop through the nodes that are in This element
			for( pos=0; pos<el->n_nodes; pos++ )
			{
				// find the node in the elements new index
				posn = P.q[el->nodes[pos]];
				
				// see if that node is connected to node in our map
				if( binary_search( node->n, node->indx , posn )<0 )
					printf( "ERROR : internal node\n" );
			}
			

		}
		node = node->next;
		i++;
	}
	 
#endif
	
	/*
	 *		go through local elements and convert all node numbering to local node order
	 */

	el   = my_elements;
	pos  = 0;
	posn = 0;
	for( i=0; i<n_els; i++ )
	{
		pos = 0;
		for( n=0; n<el->n_nodes; n++ )
		{
			// determine the new global node index
			k = P.q[el->nodes[n]];
			
			// determine and store the new local node index, checking that the index is valid
			if( k>=first && k<=last )
			{
				pos++;
				el->nodes[n] = k - first;
			}
			else
			{
				//ASSERT_MSG( (j = binary_search( n_ext, ext_list, k))>=0 , "mesh_split() : there is an inconsistancy in the mesh" );
				j = binary_search( n_ext, ext_list, k );
				
				if( j<0 )
					; //printf( "P%d - error\n", this_dom );
				
				el->nodes[n] = j + n_in + n_bnd;
			}
		}
		if( !pos )
			posn++;
		
		el++;
	}
	
	printf("\t\tP%d : %d of %d elements had no connections to This domain\n", this_dom, posn, n_els);
	
	// print stuff
	printf( "\t\tP%d/%d has %d interior, %d boundary and %d exterior nodes - and %d elements\n\n", this_dom, n_dom-1, n_in, n_bnd, n_ext, n_els );
	
	
	// find time taken in routine
	fflush( stdout );
	MPI_Barrier( This->comm );	
}

void mesh_decomp( Tmtx_CRS_dist_ptr A, Tprecon_schur_ptr P, Tnode_list_ptr *_nin, Tnode_list_ptr *_nbnd )
{
	int i, k, pos, this_dom,  n_dom, sdom, tdom, nnz_max, nodes_max, n_nodes;
	int tag, index_pos, recv_nodes, is_boundary, nnz, success, success_global;
	int n_in, n_bnd, n_node, r_pos;
	int *is_neigh, *back_counts;
	int *p, *int_temp, *nnz_pack, *tag_pack, *index_pack;
	int block, n_neigh, connect_dom;
	Tnode_ptr  node=NULL;
	Tnode_list in, bnd, nodes;
	TMPI_dat_ptr This;
	MPI_Request *request;
	MPI_Status *status;	
	
	/*
	 *		Setup the diagnostic file for This thread
	 *		
	 *		This file contains diagnostic output describing the communication
	 *		between processes/domains and the domain decomp method
	 */
	
	This = &P->This;

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

	request = malloc( 8*sizeof(MPI_Request) );
	status  = malloc( 8*sizeof(MPI_Status) );
	for( i=0; i<8; i++ )
		status[i].MPI_ERROR = MPI_SUCCESS;
	nodes_max = 0;
	for( k=0; k<n_dom; k++ )
		if( nodes_max<P->forward.counts[k] )
			nodes_max = P->forward.counts[k];
	tag_pack   = (int*)malloc( nodes_max*sizeof(int) );
	nnz_pack   = (int*)malloc( nodes_max*sizeof(int) );
	
	nnz_max    = A->mtx.nnz;
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
				
			nnz+=nnz_pack[pos];
		}
		
		//printf( "P%d : sending to P%d : %d nodes with %d indices\n", this_dom, tdom, n_nodes, nnz );		
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
		
		//printf( "P%d : receiving %d nodes from P%d\n", this_dom, recv_nodes, sdom );
		
		// check that we aren't going to run out of memory
		if( recv_nodes>nodes_max )
		{
			nodes_max = recv_nodes;
			tag_pack = (int*)realloc( tag_pack, nodes_max*sizeof(int) );
			nnz_pack = (int*)realloc( nnz_pack, nodes_max*sizeof(int) );
		}
				
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
				index_pack = (int*)realloc( index_pack, nnz_max*sizeof(int) );
			}
			
			// indices
			MPI_Irecv( index_pack,  nnz,     MPI_INT, sdom, TAG_INDEX, This->comm, request+r_pos );
			r_pos++;
		}

		//printf( "P%d : waiting for all messages to be completed\n", this_dom );
		
		// wait for all communication to finish
		for( i=0; i<r_pos; i++ )
		{
			MPI_Wait( request+i, status+i );
			if( status[i].MPI_ERROR!=MPI_SUCCESS)
				printf( "P%d -- WARNING : status for a message %d was not MPI_SUCCESS\n", this_dom, i );
		}
		
		//printf( "P%d : messages completed\n", this_dom );
		
		/*
		 *		Unpack data into node list
		 */
		for( i=0, nnz=0; i<recv_nodes; i++ )
		{
			
			// allocate memory for the new node
			if( !node_list_add( &nodes, nnz_pack[i], tag_pack[i] ) )
				fprintf( stderr, "\tP%d : ERROR, unable to add node with tag %d\n", this_dom, tag );
			
			// copy over the indices
			memcpy( nodes.opt->indx, index_pack + nnz, sizeof(int)*nnz_pack[i] );
			
			nnz += nnz_pack[i];
		}
	}
	
	free( index_pack );
	free( nnz_pack );
	free( tag_pack );
	free( request );
	free( status  );
	
	// everyone checks that they have recieved valid node lists
	success = node_list_verify( NULL, &nodes );
	MPI_Allreduce( &success, &success_global, 1, MPI_INT, MPI_LAND, This->comm );
	if( !success_global )
	{
		fprintf( stderr, "ERROR : one of the processes had an invalid node list\n" );
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
	P->p = p;
	
	// loop through the nodes
	node = in.start;
	while( node!=NULL )
	{
		node->tag = P->q[node->tag]; 
		
		// change the node indices
		for( i=0; i<node->n; i++ )
			node->indx[i] = P->q[node->indx[i]]; 
		
		// sort the indices, keeping a permutation array
		heapsort_int( node->n, node->indx);
		
		// point to next node in list
		node = node->next;
	}
	node = bnd.start;
	while( node!=NULL )
	{
		node->tag = P->q[node->tag]; 
		
		// change the node indices
		for( i=0; i<node->n; i++ )
			node->indx[i] = P->q[node->indx[i]]; 
		
		// sort the indices, keeping a permutation array
		heapsort_int( node->n, node->indx);

		// point to next node in list
		node = node->next;
	}
	
	// point the  node lists out
	*_nin  = &in;
	*_nbnd = &bnd;
	
	if( !node_list_verify( NULL, &in ) )
		printf( "P%d : ERROR : bad internal node list\n", this_dom );
	if( !node_list_verify( NULL, &bnd ) )
		printf( "P%d : ERROR : bad boundary node list\n", this_dom );
}

/* 
// first step, determine the maximum number of messages that will be passed between any two Pid
// and then allocate memory for the requests and status
msg_max = 0;
for( i=0; i<n_dom; i++ )
if( msg_max<P->forward.counts[i] )
msg_max = P->forward.counts[i];
MPI_Allreduce( &msg_max, &msg_max_global, 1, MPI_INT, MPI_MAX, This->comm );
msg_max_global = 1 + 3*msg_max_global;
request = malloc( 2*msg_max_global*sizeof(MPI_Request) );
status  = malloc( 2*msg_max_global*sizeof(MPI_Status) );

MPI_Barrier( This->comm );

for( k=0; k<n_dom; k++ )
{
	r_pos = 0;
	
	// 
	//		send out domain information
	//
	fprintf( fid, "\nk = %d\n----------\n", k );
	
	// determine the target domain
	dom = k +  this_dom;
	if( dom>=n_dom )
		dom -= n_dom;
	
	// allocate memory for MPI message requests
	n_msg_send = 3*P->forward.counts[dom] + 1;
	
	printf( "sending %d messages\n", n_msg_send );
	
	// number of nodes to send
	MPI_Isend( P->forward.counts+dom, 1,      MPI_INT, dom, TAG_NODES, This->comm, request+(r_pos++) );
	
	// only need to send data if we have some of dom's nodes
	if( P->forward.counts[dom] )
	{
		for( i=P->forward.starts[dom]; i<P->forward.starts[dom+1]; i++ )
		{
			// figure out what, where and why
			index_pos = A->mtx.rindx[P->forward.indx[i]];
			nnz = A->mtx.rindx[P->forward.indx[i]+1] - index_pos;
			indx_ptr = A->mtx.cindx + index_pos;
			tag = A->vtxdist[this_dom] + P->forward.indx[i];
			
			// now send
			MPI_Isend( &tag,     1,   MPI_INT,    dom,   TAG_TAG,   This->comm, request+(r_pos++) );
			MPI_Isend( &nnz,     1,   MPI_INT,    dom,   TAG_NNZ,   This->comm, request+(r_pos++) );
			MPI_Isend( indx_ptr, nnz, MPI_INT,    dom,   TAG_INDEX, This->comm, request+(r_pos++) );
		}
	}
	
	// 
	//	recieve domain information
	//
	
	// determine the source domain
	dom = this_dom-k;
	if( dom<0 )
		dom += n_dom;
	
	// for some reason This barrier command is needed to save MPI from getting confused
	// otherwise messages get confused, but only some of the time. It beats me why,
	// and I spent a couple of days trying to figure out why. Just let it go.
	MPI_Barrier( This->comm ); 
	
	// let the source tell me how many nodes I am to recieve
	MPI_Irecv( &recv_nodes,       1,                                 MPI_INT, dom, TAG_NODES , This->comm, request+(r_pos++) ); 
	back_counts[ dom ] = recv_nodes;
	
	// how many messages are we going to receive, includes the first receive a few lines above
	n_msg_recv = 1 + recv_nodes*3;
	
	// reveive nodes if there are any to revieve
	if( recv_nodes )
	{
		for( i=0; i<recv_nodes; i++ )
		{
			// find out the global index of This node
			MPI_Irecv( &tag,     1,   MPI_INT,    dom, TAG_TAG,   This->comm, request+(r_pos++)  );
			
			// find out how many nnz are associated with This node
			MPI_Irecv( &nnz,     1,   MPI_INT,    dom, TAG_NNZ,   This->comm, request+(r_pos++) );
			if( nnz>max_node_nnz )
				max_node_nnz = nnz;
			
			
			// create new node, checking for any errors. If there is an error, the loop continues, and
			// lets node_list_verify() find and analyze the problem after all communication is finished, allowing all the processes
			// to abort in an organised and orderly manner.
			if( !node_list_add( &nodes, nnz, tag ) )
				fprintf( stderr, "\tP%d : ERROR, unable to add node with tag %d\n", this_dom, tag );
			
			// receive the new node data
			MPI_Irecv( nodes.opt->indx, nnz, MPI_INT,    dom, TAG_INDEX, This->comm, request+(r_pos++) );
		}
	}
	
	printf( "r_pos = %d, when it should be %d\n", r_pos, 2 + 3*(n_msg_send+n_msg_recv) );
	
	//  wait till all messages have been sent/received
	MPI_Waitall( n_msg_send+n_msg_recv, request, status );
	
	// wait until all the data I have sent is recieved
	MPI_Barrier( This->comm );
}

// free up memory used for message passing request/status
free( status );
free( request ); */

