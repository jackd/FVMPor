/*
 *  split.c
 *  
 *
 *  Created by Ben Cumming on 15/06/05.
 *  Copyright 2005 __MyCompanyName__. All rights reserved.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <scsl_blas.h>
#include <unistd.h>
#include <mpi.h>

#include "linalg_dense.h"
#include "linalg_sparse.h"
#include "linalg.h"
#include "fileio.h"
#include "benlib.h"
#include "linalg_mpi.h"
#include "ben_mpi.h"
#include "parmetis.h"
#include "indices.h"
#include "gmres.h"
#include "MR.h"
#include "precon.h"
#include "ILU.h"
#include "domain_decomp.h"
#include "metis.h"

/******************************************************************************************
*		This isn't all that efficient, however it is robust, and it gets the job done
*		in a reasonable time... and besides, we only have to do it once for any given mesh
******************************************************************************************/
void main( int argc, char *argv[] )
{
	int i, j, n_doms, this_dom, count;
	int n_node_props, n_element_props;
	TMPI_dat This;
	Tmesh mesh;
	Tsplit split;
	double time;
	int *vtxdist, *part_global, *part_local, *cpy;
	int *nodes_int, *nodes_bnd, *nodes_local, *nodes_ext;
	int n_nodes_int, n_nodes_ext, n_nodes_bnd, n_nodes_local, n_nodes_global, n_elements, n_elements_bnd, n_elements_int;
	int element, node, pos, internal, ext_array_size;
	int *local_node_indices, *local_element_indices, *ppos, *elements_flag, *elements_local, *p, *q;
	double *element_props;
	char fname[100];
	char *fname_mesh, *fname_props, *fname_perm, *fname_out;
	Telement_ptr el;
	FILE *fid;
	
	/*
	 *		begin MPI
	 */
	BMPI_init( argc, argv, &This );
	this_dom = This.this_proc;
	
	// let everyone know that This isn't Kansas
	if( !This.this_proc )
		printf( "\n----------------------------------------------------------\nDomain splitting\n----------------------------------------------------------\n\n" );

	/*
	 *		load the compiled (complete) mesh from disk
	 */
	
	if( argc!=5 )
	{
		if( !This.this_proc )
		{
			printf( "ERROR : wrong number of parameters on command line - argc = %d\n\n", argc );
			printf( "use   : \tmpirun -np N ./split meshfname    propname    permname     outname \n\neg    :    \tmpirun -np 8 ./split global.mesh  global.dat  global.perm  global\n" );
			printf( "\nwhere : \tglobal.dat contains the properties information for the nodes and elements\n" );
			printf( "\t\tglobal.mesh and global.perm are the compiled output from decomp\n\t\tand global is the stub for the .pmesh and .pprop files\n\n" );
		}

		MPI_Finalize();
		exit(1);
	}
		
	fname_mesh = argv[1];
	fname_props = argv[2];
	fname_perm = argv[3];
	fname_out = argv[4];
	
	MPI_Barrier( This.comm );
	if( !This.this_proc )
	{
		printf( "loading mesh...\n" );
		time = get_time();
	}
	if( !mesh_compiled_read( &mesh, fname_mesh ) )
	{
		printf( "P%d : ERROR loading from file %s\n", This.this_proc, fname );
		MPI_Finalize();
		return;
	}
	MPI_Barrier( This.comm );
	if( !This.this_proc )
	{
		printf( "\tmesh loaded with %d nodes and %d elements\n", mesh.n_nodes, mesh.n_elements );
		time = get_time() - time;
		printf( "\tmesh loaded in %g seconds\n", time );		
	}
	
	/************************
	 * mesh verify here?
	 ************************/
	
	/*
	 *		load up the global permutation vector
	 */
	if( !This.this_proc ) printf( "\nloading domain decomposition permutation from %s...\n", fname_perm );
	
	n_nodes_global = mesh.n_nodes;
	part_global = (int*)malloc( n_nodes_global*sizeof(int) );
	vtxdist     = (int*)malloc( (This.n_proc+1)*sizeof(int)  );
	if( !( fid = fopen( fname_perm, "rb" ) ) )
	{
		if( !this_dom )
			printf( "\tERROR : unable to load permutation file for input, exitting\n\n" );
		MPI_Finalize();
		return;
	}
	
	fread( &n_doms, sizeof(int), 1, fid );
	if( n_doms!=This.n_proc )
	{
		if( !this_dom ) printf( "\n\nERROR : the number of domains must equal the number of Pid (n_dom = %d, n_proc = %d)\n", n_doms, This.n_proc );
		fclose( fid );
		MPI_Finalize();
		return;
	}
	fread( vtxdist, sizeof(int), n_doms+1, fid );
	fread( part_global, sizeof(int), n_nodes_global, fid );		
	fclose( fid );
	
	
	/*
	 *		run through the vector and draw out a list of all the nodes that I own
	 */
	part_local = malloc( sizeof(int)*n_nodes_global );
	for( i=0, count=0; i<n_nodes_global; i++ )
	{
		if( part_global[i]==this_dom )
		{
			part_local[count] = i;
			count++;
		}
	}
	n_nodes_local = count;
	
	part_local = realloc( part_local, sizeof(int)*n_nodes_local );	
	
	/*
	 *		now to determine which nodes are internal and boundary - and determine the external nodes
	 *		belonging to other domains
	 */
	
	if( !This.this_proc ) printf( "\nclassifying internal and boundary nodes...\n" );
	
	// allocate memory
	nodes_int = (int*)malloc( sizeof(int)*n_nodes_local );
	nodes_bnd = (int*)malloc( sizeof(int)*n_nodes_local );
	nodes_ext = (int*)malloc( sizeof(int)*n_nodes_local );
	ext_array_size = n_nodes_local;
	n_nodes_int = n_nodes_ext = n_nodes_bnd = 0;
	
	// loop through each node in the local domain and determine
	// if it is local or boundary
	for( i=0; i<n_nodes_local; i++ )
	{
		internal = 1;
		for( j=mesh.A.rindx[part_local[i]]; j<mesh.A.rindx[part_local[i]+1] ; j++ )
		{
			if( part_global[mesh.A.cindx[j]]!=this_dom )
			{
				// flag that This node is a boundary node				
				internal = 0;
				
				// add the exterior node to which the node is connected to the external node lost
				if( n_nodes_ext>=ext_array_size-2 )
				{
					ext_array_size += n_nodes_local;
					nodes_ext = (int*)realloc( nodes_ext, sizeof(int)*ext_array_size );
				}
				nodes_ext[ n_nodes_ext++ ] = mesh.A.cindx[j];
			}
		}
		
		// is the node internal?
		if( internal )
		{
			// it is so add it to the internal list
			nodes_int[ n_nodes_int++ ] = part_local[i];
		}
		else
		{
			// it must be a boundary node so add it to the boundary list
			nodes_bnd[ n_nodes_bnd++ ] = part_local[i];
		}
	}
		
	// reallocate memory to reflect the number of internal and boundary nodes
	nodes_int = (int*)realloc( nodes_int, sizeof(int)*n_nodes_int );
	nodes_bnd = (int*)realloc( nodes_bnd, sizeof(int)*n_nodes_bnd );
	
	// sort the external nodes
	heapsort_int( n_nodes_ext, nodes_ext );
	
	// determine the unique external indices
	nodes_ext[n_nodes_ext] = -1;
	for( j=0, i=0; i<n_nodes_ext; j++, i++ )
	{
		while( nodes_ext[i]==nodes_ext[i+1] )
			i++;
		nodes_ext[j] = nodes_ext[i];
	}
	n_nodes_ext = j;
	nodes_ext = (int*)realloc( nodes_ext, sizeof(int)*n_nodes_ext );
	
	cpy = (int*)malloc( sizeof(int)*n_nodes_ext );
	memcpy( cpy, nodes_ext, sizeof(int)*n_nodes_ext );
	
	// make a list of all the local nodes with the external nodes appended
	nodes_local = (int*)malloc(  sizeof(int)*(n_nodes_local + n_nodes_ext) );
	memcpy( nodes_local,             nodes_int, sizeof(int)*n_nodes_int );
	memcpy( nodes_local+n_nodes_int, nodes_bnd, sizeof(int)*n_nodes_bnd );
	memcpy( nodes_local+n_nodes_local, nodes_ext, sizeof(int)*n_nodes_ext );	

	
	/*
	 *		Setup the split data structure
	 */
	BMPI_copy( &This, &split.This );
	domain_init( &split.dom, &This, n_nodes_global );
	split.counts = (int*)malloc( n_doms*sizeof(int) );
	split.n_int = n_nodes_int;
	split.n_bnd = n_nodes_bnd;
	split.n_ext = n_nodes_ext;
	split.nodes_ext = (int*)malloc( n_nodes_ext*sizeof(int) );
	
	/*
	 *		Now all Pid communicate with one another to determine the new global order
	 *
	 *		we need This information so that each domain knows where its external nodes
	 *		now lie in the grand scheme of things
	 */
	
	if( !This.this_proc ) printf( "\ncommunicating global permutation data...\n" );
	
	// determine what the counts and vtxdist are
	MPI_Allgather( &n_nodes_local, 1, MPI_INT, split.counts, 1, MPI_INT, This.comm  );
	split.dom.vtxdist[0] = 0;
	for( i=0; i<n_doms; i++ )
		split.dom.vtxdist[i+1] = split.dom.vtxdist[i] + split.counts[i];	
	
	// allocate memory for temporary working arrays
	p = (int *)malloc( sizeof(int)*n_nodes_global );
	q = (int *)malloc( sizeof(int)*n_nodes_global );
		
	// gather each domain's local ordering for the original node tags
 	MPI_Allgatherv( nodes_local, n_nodes_local, MPI_INT, p, split.counts, split.dom.vtxdist, MPI_INT, This.comm );
	
	// convert into global permutation
	for( i=0; i<n_nodes_global; i++ )
		q[p[i]] = i;
	
	if( This.this_proc )
	{
		sprintf( fname, "%s_%d.prm", fname_out, n_doms );
		fid = fopen( fname, "wb" );
		fwrite( q, sizeof(int), n_nodes_global, fid );
		fread( vtxdist, sizeof(int), n_doms+1, fid );
		fclose( fid );
	}
	
	/*
	 *			Make a list of all the elements that are associated with This subdomain
	 *
	 *			(1) for each node add all of the elements it is connected to
	 *			(2) sort and make the list of elements unique
	 *			(3) for each element sort it's nodes and flag any elements that have external nodes
	 *			    as boundary elements.
	 */
	
	if( !This.this_proc ) printf( "\ndetermining local elements...\n" );
	
	// allocate enough room for 5 times as many elements as nodes
	ext_array_size = n_nodes_local*5;
	elements_local = (int*)malloc( sizeof(int)* ext_array_size );
	
	// loop over each node.
	// note that we do not have to look at external nodes since
	// the relevant element will be connected to a local node
	// if we check external nodes then elements that are internal to
	// another domain may be introduced. That would be bad.
	for( i=0, pos=0; i<n_nodes_local; i++ )
	{
		node = nodes_local[i];
		n_elements = mesh.node_num_elements[node];
		
		// check that we won't overrun the end of the array
		if( (pos+n_elements) > (ext_array_size-2) )
		{
			// we have so allocate some more memory
			ext_array_size += 3*n_elements*(n_nodes_local-i);
			elements_local = (int*)realloc( elements_local, sizeof(int)*ext_array_size );
		}
		
		for( j=0; j<n_elements; j++, pos++ )
		{
			elements_local[pos] = mesh.node_elements[node][j];
		}
	}
	
	// sort the elements and find a unique order
	heapsort_int( pos, elements_local );
	
	elements_local[pos] = -1;
	for( j=0, i=0; i<pos; j++, i++ )
	{
		while( elements_local[i]==elements_local[i+1] )
			i++;
		elements_local[j] = elements_local[i];
	}
	n_elements = j;
	elements_local = (int*)realloc( elements_local, sizeof(int)*n_elements );
		
	// search each element for external connections
	// if there is an external connection then it is labelled 1, internal labelled 0
	// This might seem counterintuative but it is used for sorting the elements
	elements_flag = (int*)malloc( sizeof(int)*n_elements );
	n_elements_int = 0;
	for( i=0; i<n_elements; i++ )
	{
		internal = 1;
		element = elements_local[i];
		for( j=0; j<mesh.elements[element].n_nodes; j++ )
		{
			if( part_global[mesh.elements[element].nodes[j]]!=this_dom )
				internal = 0;
		}

		elements_flag[i] = 1-internal;
		n_elements_int += internal;
	}
	n_elements_bnd = n_elements - n_elements_int;
	heapsort_int_index( n_elements, elements_local, elements_flag );
	heapsort_int( n_elements_int, elements_local );
	heapsort_int( n_elements_bnd, elements_local + n_elements_int );
		
	/*
	 *		make all of the nodes' and elements' references to one another in the new local ordering
	 */
	
	if( !This.this_proc ) printf( "\nlocalising node-element references...\n" );
	
	// create a lookup array that gives the local node order for a node in the original 
	// global ordering. If a node is not local or external for This domain lookup is -1
	local_node_indices = (int*)malloc( sizeof(int)*n_nodes_global );
	for( i=0; i<n_nodes_global; i++ )
		local_node_indices[i] = -1;
	for( i=0; i<n_nodes_local+n_nodes_ext; i++ )
		local_node_indices[nodes_local[i]] = i;
	
	// loop over each of the elements and change the node numbers to new local order
	for( i=0; i<n_elements; i++ )
	{
		element = elements_local[i];
		for( j=0, ppos=mesh.elements[element].nodes; j<mesh.elements[element].n_nodes; j++, ppos++ )
		{
			*ppos = local_node_indices[*ppos];
			
			// check that the node actually is local or external to This domain
			// This is a bug check
			ASSERT_MSG( *ppos!=-1, "An node in a local element is neither local or external to This domain" );
		}
	}
	
	// make all nodes' references to elements local
	local_element_indices = (int*)malloc( sizeof(int)*mesh.n_elements );
	for( i=0; i<mesh.n_elements ; i++ )
		local_element_indices[i] = -1;
	for( i=0; i<n_elements; i++ )
		local_element_indices[elements_local[i]] = i;
	
	// do local nodes first
	for( i=0; i<n_nodes_local; i++ )
	{
		node = nodes_local[i];
		for( j=0; j<mesh.node_num_elements[node]; j++ )
		{
			element = local_element_indices[mesh.node_elements[node][j]];
			ASSERT_MSG( element!=-1, "Invalid element reference." );
			mesh.node_elements[node][j] = element;
		}
	}
	
	// then exterior nodes, taking care to only include local elements
	for( i=n_nodes_local; i<(n_nodes_local+n_nodes_ext); i++ )
	{
		node = nodes_local[i];
		for( pos=0, j=0; j<mesh.node_num_elements[node]; j++ )
		{
			element = local_element_indices[mesh.node_elements[node][j]];
			if( element!=-1 )
				mesh.node_elements[node][pos++] = element;
		}
		mesh.node_num_elements[node] = pos;
	}
	
	/*
	 *		Output the data
	 */
	
	// output some statistics to the screen
	if( !this_dom )
	{
		printf( "\nsubdomain mesh statistics\n=========================\n\n" );
	}
	for( i=0; i<n_doms; i++ )
	{
			fflush(stdout);
			MPI_Barrier( This.comm );
			if( i==this_dom )
			{
				printf( "\tP%d : %d internal nodes, %d boundary nodes and  %d external nodes\n", this_dom, n_nodes_int, n_nodes_bnd, n_nodes_ext );
				printf( "\tP%d : %d local elements from %d global elements of which %d/%d are internal/boundary\n\n", this_dom, n_elements, mesh.n_elements, n_elements_int, n_elements_bnd );
			}
			time = get_time();
			while( (get_time()-time)<0.01 );
	}

	// let the viewer know the name of the output file
	if( !this_dom )
	{
		printf( "\noutputting subdomain mesh data...\n" );
		sprintf( fname, "%s_%d_X.pmesh", fname_out, n_doms );
		printf( "\toutputting to %s, where X is the subdomain number\n", fname );
	}
	 
	// determine the filename for output
	sprintf( fname, "%s_%d_%d.pmesh", fname_out, n_doms, this_dom );

	// open the file
	fid = fopen( fname, "wb" );
	
	// write the data
	fwrite( &This.n_proc,      sizeof(int), 1 , fid );
	fwrite( &This.this_proc,   sizeof(int), 1 , fid );
	fwrite( split.dom.vtxdist, sizeof(int), This.n_proc+1 , fid );
	fwrite( &n_nodes_global,   sizeof(int), 1 , fid );
	fwrite( &n_nodes_int,      sizeof(int), 1 , fid );
	fwrite( &n_nodes_bnd,      sizeof(int), 1 , fid );
	fwrite( &n_nodes_ext,      sizeof(int), 1 , fid );
	fwrite( &n_elements_int,   sizeof(int), 1 , fid );
	fwrite( &n_elements_bnd,   sizeof(int), 1 , fid );
	for( i=0; i<n_nodes_ext; i++ )
		nodes_ext[i] = q[nodes_ext[i]]; 
	fwrite( nodes_ext,         sizeof(int), n_nodes_ext , fid );
	for( i=0; i<n_nodes_local + n_nodes_ext; i++ )
	{
		node = nodes_local[i];
		fwrite( &mesh.node_x[node], sizeof(double), 1 , fid );
		fwrite( &mesh.node_y[node], sizeof(double), 1 , fid );
		fwrite( &mesh.node_z[node], sizeof(double), 1 , fid );
		fwrite( &mesh.node_bc[node], sizeof(int), 1 , fid );
	}
	for( i=0; i<n_elements; i++ )
	{
		el = mesh.elements + elements_local[i];
		fwrite( &el->n_nodes, sizeof(int), 1 , fid );
		fwrite( &el->n_edges, sizeof(int), 1 , fid );
		fwrite( &el->n_faces, sizeof(int), 1 , fid );
		fwrite( el->nodes, sizeof(int)*el->n_nodes, 1 , fid );
		fwrite( el->face_bcs, sizeof(int)*el->n_faces, 1 , fid );
	}
	
	// close the file
	fclose( fid );
	
	/*
	 *		Now load and distribute the node/element properties
	 */
	
	// load
	fid = fopen( fname_props, "rb" );
	if( fid==NULL )
	{
		if( !this_dom )
			fprintf( stderr, "ERROR : unable to open properties file %s for input\n", fname_props );
		MPI_Finalize();
		return;
	} 
	fread( &n_node_props, sizeof(int), 1, fid );
	fread( &n_element_props, sizeof(int), 1, fid );
	
	if( n_node_props!=0 || n_element_props!=2 )
	{
		if( !this_dom )
			fprintf( stdout, "ERROR : I don't know how to handle %d node properties and %d element properties in %s\n", n_node_props, n_element_props, argv[4]  );
		MPI_Finalize();
		return;
	}
	
	element_props = (double*)malloc( sizeof(double)*n_element_props*mesh.n_elements );
	fread( element_props, sizeof(double), n_element_props*mesh.n_elements, fid );
	fclose( fid );
	
	// write
	sprintf( fname, "%s_%d_%d.pprop", fname_out, n_doms, this_dom );
	fid = fopen( fname, "wb" );
	
	fwrite( &n_node_props, sizeof(int), 1, fid );
	fwrite( &n_element_props, sizeof(int), 1, fid );
	for( i=0; i<n_elements; i++ )
	{
		element = elements_local[i];
		fwrite( element_props + element*n_element_props, sizeof(double), n_element_props, fid );
	}
	fclose( fid );
	
	// close down MPI	
	MPI_Finalize();
}

