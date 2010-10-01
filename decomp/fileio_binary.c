#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linalg_sparse.h"
#include "linalg_dense.h"
#include "fileio.h"
#include "domain_decomp.h"
#include "linalg_mpi.h"

int mesh_load( Tmesh_ptr mesh, char *fname )
{
	int i, j, k, n_nodes, n_elements, node, max_el, pos, nnz, nnz_max;
	double *node_x, *node_y, *node_z;
	int  *node_num_elements, *node_list, *node_bc;
	int **node_elements;
	Telement_ptr elements, el;
	FILE *fid;
	Tmtx_CRS_ptr A;
	
        fprintf(stderr, "\tERROR : mesh_load() binary version needs updating for multiple BCs on a node\n");
        error(1);

	// setup various bits'n peices
	A = &mesh->A;
	
	A->init = 0;
	
	// open file in binary
	if( !(fid=fopen(fname,"rb")) )
	{
		fprintf( stderr, "\tERROR : mesh_read() : unable to open file %s for input\n", fname );
		return 0;
	}

	// read in the size of the mesh
	fread( &n_nodes, sizeof(int), 1, fid );
	fread( &n_elements, sizeof(int), 1, fid );
	
	printf( "%d nodes and %d elements\n", n_nodes, n_elements );
	
	// allocate memory for nodes
	node_x = (double*)malloc( n_nodes*sizeof(double) );
	node_y = (double*)malloc( n_nodes*sizeof(double) );
	node_z = (double*)malloc( n_nodes*sizeof(double) );
	//node_bc = (int*)malloc( n_nodes*sizeof(int) );
	node_num_elements = (int*)calloc( n_nodes, sizeof(int) );

	// read in coordinates of nodes
	for( i=0; i<n_nodes; i++ )
	{
		fread( node_x+i, sizeof(double), 1, fid );
		fread( node_y+i, sizeof(double), 1, fid );
		fread( node_z+i, sizeof(double), 1, fid );
		//fread( node_bc+i, sizeof(int), 1, fid );
	}
	
	// allocate memory for elements
	elements = (Telement_ptr)malloc( n_elements*sizeof(Telement) );
	el = elements;
	

	// read in the element data
	for( i=0; i<n_elements; i++ )
	{

		// read in basic data about the element
		fread( &el->n_nodes, sizeof(int), 1, fid );
		fread( &el->n_edges, sizeof(int), 1, fid );
		fread( &el->n_faces, sizeof(int), 1, fid );
				

		// allocate memory for the node and face data for This element
		el->nodes    = (int*)malloc( el->n_nodes*sizeof(int) );
		el->face_bcs = (int*)malloc( el->n_faces*sizeof(int) );
		
		// scan in the node and face data
		fread( el->nodes,    sizeof(int), el->n_nodes, fid );
		
		// we don't read in the face BC data as it isn't in the file yet, though it will  be. instead
		// I replace it with a the element number, which allows us to check that the code works
		for( k=0; k<el->n_faces; k++ )
			el->face_bcs[k] = i;
		fread( el->face_bcs, sizeof(int), el->n_faces, fid );

		// validate the input information
		for( j=0; j<el->n_nodes; j++ )
		{
			node_num_elements[el->nodes[j]]++;
			if( el->nodes[j]<0 || el->nodes[j]>=n_nodes )
			{
				fprintf( stderr, "ERROR : mesh_read() : invalid element->node data\n" );
				return 0;
			}
		}
					
		// point to the next element
		el++;
	}

	/*
	 *		find out how many elements each node is in
	 */
	node_elements = (int**)malloc( n_nodes*sizeof(int*) );
	for( i=0; i<n_nodes; i++ )
	{
		node_elements[i] = (int*)malloc( node_num_elements[i]*sizeof(int) );
		node_num_elements[i] = 0;
	}

	/*
	 *		find out which nodes each node is connected to
	 */
	
	// determine the set of elements that each node belongs to
	el = elements;
	for( i=0; i<n_elements; i++ )
	{
		for( j=0; j<el->n_nodes; j++ )
		{	
			node = el->nodes[j];
			node_elements[node][node_num_elements[node]++] = i;
		}
		
		el++;
	}
	
	// determine the maximum number of elements that a node belongs to
	// also gather an upper bound on the number of nz in the Jacobian
	max_el = 0;
	nnz_max = 0;
	for( i=0; i<n_nodes; i++ )
	{
		if( node_num_elements[i]>max_el )
			max_el = node_num_elements[i];
		nnz_max += node_num_elements[i];
	}
	nnz_max *= 7;
	
	// allocate CRS matrix for storing the pattern
	mtx_CRS_init( A, n_nodes, n_nodes, nnz_max, 0 );
	free( A->nz );
	A->nz = NULL;
	
	/*
	 *		make the CRS map
	 */
	node_list = (int*)malloc( 8*max_el*sizeof(int) );
	nnz = 0;
	A->rindx[0] = 0;
	for( node=0; node<n_nodes; node++ )
	{
		// make list of all nodes This node is connected to
		pos = 0;
		for( j=0; j<node_num_elements[node]; j++ )
		{
			el = elements + node_elements[node][j];
			
			for( i=0; i<el->n_nodes; i++ )
				node_list[pos++] = el->nodes[i];
		}
	
		// sort the list
		heapsort_int( pos, node_list );
		
		// remove duplicates
		j = node_list[0];
		k = 1;
		for( i=1; i<pos; i++ )
		{
			if( j!=node_list[i] )
			{
				node_list[k] = j = node_list[i];
				k++;
			}
		}
		
		// store in CRS form
		nnz += k;
		A->rindx[node+1] = nnz;
		for( i=A->rindx[node], pos = 0; pos<k; i++, pos++ )
		{
			A->cindx[i] = node_list[pos];
		}
	}
	
	A->nnz = nnz;
	A->cindx = (int*)realloc( A->cindx, sizeof(int)*nnz );
	
	// give all of the elements tag numbers. These may be used by the parallel code
	// to check for internal consistency of the mesh
	for( i=0; i<n_elements; i++ )
	{
		elements[i].tag = i;
	}
	
	// free memory
	fclose( fid );
	free( node_list );

	mesh->node_elements = node_elements;
	mesh->node_num_elements = node_num_elements;
	mesh->n_elements = n_elements;
	mesh->elements = elements;
	mesh->n_nodes  = n_nodes;
	mesh->node_x   = node_x;
	mesh->node_y   = node_y;
	mesh->node_z   = node_z;
	//mesh->node_bc  = node_bc;
	
	return 1;
}

/*
int mtx_CCS_load( char *fname, Tmtx_CCS_ptr mtx )

load a sparse matrix from matrix market sparse format.
all matrics are loaded into an unsymmetric format.

PRE  fname is a string containing the name of the file
     mtx is uninitialised
POST returns 1 => mtx is succesfully initialised and loaded
     returns 0 => unable to load the matrix, mtx is uninitialised
*/
int mtx_CCS_load( char *fname, Tmtx_CCS_ptr mtx )
{
  FILE *fin;
  char header[200];
  int col, pos, ctemp, ncols, nrows, nnz;

  /* open the file */
  if( (fin=fopen(fname, "r"))==NULL )
  {
    fprintf( stdout, "\n\nERROR: unable to load the file %s\n\n", fname );
    return 0;
  }

  /* read the header */
  fscanf( fin, "%s",header );
  fscanf( fin, "%s",header );
  fscanf( fin, "%s",header );
  fscanf( fin, "%s",header );
  fscanf( fin, "%s",header );
  ASSERT_MSG( strcmp( header, MTXSYMMETRIC ), "mtx_CCS_load() : unable to handle symmetric matrices at This point" )

  if( !strcmp( header, MTXGENERAL ) )
  {
	  float ftemp;
	  
	  /* read in the matrix stats */
	  fscanf( fin, "%d %d %d", &(nrows), &(ncols), &(nnz) );
	  
	  if( nrows<=0 || ncols<= 0  )
	  {
		  fprintf( stderr, "WARNING : invalid matrix dimensions found when reading matrix from file %s\n", fname );
		  fprintf( stderr, "\t\tdimensions %dX%d with %d nz\n", nrows, ncols, nnz );
		  fclose( fin );
		  return 0;
	  }
	  
	  /* initialise the matrix */
	  mtx_CCS_init( mtx, nrows, ncols, nnz, 0 );
	  
	  /* save the nz entries */
	  mtx->cindx[0]=0;
	  col=1;
	  for( pos=0; pos<mtx->nnz; pos++ )
	  {
		  fscanf( fin, "%d %d %f", &(mtx->rindx[pos]), &ctemp, &ftemp );
		  if( ctemp>col )
		  {
			  while( col<ctemp )
				  mtx->cindx[col++]=pos;
		  }
		  mtx->rindx[pos]--;
		  
		  mtx->nz[pos]=(double)ftemp; 
	  }
	  
	  /* add the false column at the end */
	  mtx->cindx[mtx->ncols]=mtx->nnz;
  }
  else if( !strcmp( header, MTXGENERALBLOCK ) )
  {
	  float ftemp[4];
	  double *bpos;
	  
	  /* read in the matrix stats */
	  fscanf( fin, "%d %d %d", &(nrows), &(ncols), &(nnz) );
	  
	  /* initialise the matrix */
	  mtx_CCS_init( mtx, nrows, ncols, nnz, 1 );
	  
	  /* save the nz entries */
	  mtx->cindx[0] = 0;
	  col = 1;
	  bpos = mtx->nz;
	  for( pos=0; pos<mtx->nnz; pos++, bpos+=(1 BLOCK_M_SHIFT) )
	  {
		  fscanf( fin, "%d %d %f %f %f %f", &(mtx->rindx[pos]), &ctemp, ftemp, ftemp+1, ftemp+2, ftemp+3 );
		  if( ctemp>col )
		  {
			  while( col<ctemp )
				  mtx->cindx[col++]=pos;
		  }
		  mtx->rindx[pos]--;
		  
		  // store the block
		  BLOCK_M_COPY( ftemp,  bpos );
	  }
	  
	  /* add the false column at the end */
	  mtx->cindx[mtx->ncols]=mtx->nnz;
  }
  else
  {
	  fprintf( stdout, "\n\nERROR : unable to understand header in %s\n\n", fname );
	  return 0;
  }

  fclose( fin );
  
  /* return success */
  return 1;
}

/*
 int mtx_CCS_load( char *fname, Tmtx_CCS_ptr mtx )
 
 load a sparse matrix from matrix market sparse format.
 all matrics are loaded into an unsymmetric format.
 
 PRE  fname is a string containing the name of the file
 mtx is uninitialised
 POST returns 1 => mtx is succesfully initialised and loaded
 returns 0 => unable to load the matrix, mtx is uninitialised
 */
int mtx_CRS_load( char *fname, Tmtx_CRS_ptr mtx )
{
	FILE *fin;
	char header[200];
	int row, pos, ctemp, ncols, nrows, nnz, col, rtemp;
	
	/* open the file */
	if( (fin=fopen(fname, "r"))==NULL )
	{
		fprintf( stdout, "\n\nERROR: unable to load the file %s\n\n", fname );
		return 0;
	}
	
	/* read the header */
	fscanf( fin, "%s",header );
	fscanf( fin, "%s",header );
	fscanf( fin, "%s",header );
	fscanf( fin, "%s",header );
	fscanf( fin, "%s",header );
	ASSERT_MSG( strcmp( header, MTXSYMMETRIC ), "mtx_CRS_load() : unable to handle symmetric matrices at This point" );
		
	if( !strcmp( header, MTXGENERAL ) )
	{
		float ftemp;
		
		/* read in the matrix stats */
		fscanf( fin, "%d %d %d", &(nrows), &(ncols), &(nnz) );
		
		if( nrows<=0 || ncols<= 0  )
		{
			fprintf( stderr, "WARNING : invalid matrix dimensions found when reading matrix from file %s\n", fname );
			fprintf( stderr, "\t\tdimensions %dX%d with %d nz\n", nrows, ncols, nnz );
			fclose( fin );
			return 0;
		}
		
		/* initialise the matrix */
		mtx_CRS_init( mtx, nrows, ncols, nnz, 0 );
		
		/* save the nz entries */
		mtx->rindx[0]=0;
		row=1;
		for( pos=0; pos<mtx->nnz; pos++ )
		{
			fscanf( fin, "%d %d %f", &rtemp, &col, &ftemp );
			if( rtemp>row )
			{
				while( row<rtemp )
					mtx->rindx[row++]=pos;
			}
			mtx->cindx[pos] = col-1;
			
			mtx->nz[pos]=(double)ftemp; 
		}
		
		/* add the false column at the end */
		mtx->rindx[mtx->nrows]=mtx->nnz;
	}
	else if( !strcmp( header, MTXGENERALBLOCK ) )
	{
		float ftemp[4];
		double *bpos;
		
		/* read in the matrix stats */
		fscanf( fin, "%d %d %d", &(nrows), &(ncols), &(nnz) );
		
		/* initialise the matrix */
		mtx_CRS_init( mtx, nrows, ncols, nnz, 1 );
		
		/* save the nz entries */
		mtx->rindx[0] = 0;
		row = 1;
		bpos = mtx->nz;
		for( pos=0; pos<mtx->nnz; pos++, bpos+=(1 BLOCK_M_SHIFT) )
		{
			fscanf( fin, "%d %d %f %f %f %f", &ctemp, &(mtx->cindx[pos]), ftemp, ftemp+1, ftemp+2, ftemp+3 );
			if( ctemp>row )
			{
				while( row<ctemp )
					mtx->rindx[row++]=pos;
			}
			mtx->cindx[pos]--;
			
			// store the block
			BLOCK_M_COPY( ftemp,  bpos );
		}
		
		/* add the false column at the end */
		mtx->rindx[mtx->nrows]=mtx->nnz;
	}
	else
	{
		fprintf( stdout, "\n\nERROR : unable to understand header in %s\n\n", fname );
		return 0;
	}
	
	fclose( fin );
	
	/* return success */
	return 1;
}



/*
 output a matrix in matrix market format 

 you should not need any explanation
*/
int mtx_CCS_output( Tmtx_CCS_ptr A, char *fname )
{
  FILE *fid;
  int i, col;

  if( (fid=fopen( fname, "w"))==NULL )
  {
    fprintf( stdout, "ERROR : Unable to open file %s for the output of matrix data\n", fname );
    return 0;
  }

  if( A->block )
	  fprintf( fid, "%%%%MatrixMarket matrix coordinate real generalblock\n" );  
  else
	  fprintf( fid, "%%%%MatrixMarket matrix coordinate real general\n" );
  fprintf( fid, "%d %d %d\n", A->nrows, A->ncols, A->nnz );
  if( A->nz )
  {
	  if( A->block )
	  {
		  double *nzp = A->nz;
		  
		  for( col=1, i=0; i<A->nnz; i++, nzp+=(BLOCK_SIZE*BLOCK_SIZE) )
		  {
			  while( i>=A->cindx[col])
				  col++;
			  fprintf( fid, "%d %d %20g %20g %20g %20g\n", A->rindx[i]+1, col,  nzp[0],  nzp[1],  nzp[2],  nzp[3] );
		  }
	  }
	  else
	  {
		  for( col=1, i=0; i<A->nnz; i++ )
		  {
			  if( i>=A->cindx[col])
				  col++;
			  fprintf( fid, "%d %d %20g\n", A->rindx[i]+1, col,  A->nz[i] );
		  }
	  }
  }
  else // pattern matrix
  {
	  	  for( col=1, i=0; i<A->nnz; i++ )
		  {
			  if( i>=A->cindx[col])
				  col++;
			  fprintf( fid, "%d %d 1\n", A->rindx[i]+1, col );
		  }
  }

  fclose( fid );

  return 1;
}

/*
 output a matrix in matrix market format 
 
 you should not need any explanation
 */
int mtx_CCS_output_matlab( Tmtx_CCS_ptr A, char *fname )
{
	FILE *fid;
	int i, col;
	
	if( (fid=fopen( fname, "w"))==NULL )
	{
		fprintf( stdout, "ERROR : Unable to open file %s for the output of matrix data\n", fname );
		return 0;
	}
	
	if( A->block )
		fprintf( fid, "%d %d %d 0 0 0\n1 0 0 0 0 0\n", A->nrows, A->ncols, A->nnz );
	else
		fprintf( fid, "%d %d %d\n0 0 0\n", A->nrows, A->ncols, A->nnz );
	if( A->nz )
	{
		if( A->block )
		{
			double *nzp = A->nz;
			
			for( col=1, i=0; i<A->nnz; i++, nzp+=(BLOCK_SIZE*BLOCK_SIZE) )
			{
				while( i>=A->cindx[col])
					col++;
				fprintf( fid, "%d %d %20g %20g %20g %20g\n", A->rindx[i]+1, col,  nzp[0],  nzp[1],  nzp[2],  nzp[3] );
			}
		}
		else
		{
			for( col=1, i=0; i<A->nnz; i++ )
			{
				if( i>=A->cindx[col])
					col++;
				fprintf( fid, "%d %d %20g\n", A->rindx[i]+1, col,  A->nz[i] );
			}
		}
	}
	else // pattern matrix
	{
		for( col=1, i=0; i<A->nnz; i++ )
		{
			if( i>=A->cindx[col])
				col++;
			fprintf( fid, "%d %d 1\n", A->rindx[i]+1, col );
		}
	}
	
	fclose( fid );
	
	return 1;
}


/*
 output a matrix in matrix market format 
 
 you should not need any explanation
 */
int mtx_CRS_output( Tmtx_CRS_ptr A, char *fname )
{
	FILE *fid;
	int i, row;
	
	if( (fid=fopen( fname, "w"))==NULL )
	{
		fprintf( stdout, "ERROR : Unable to open file %s for the output of matrix data\n", fname );
		return 0;
	}
	
	if( A->block )
		fprintf( fid, "%%%%MatrixMarket matrix coordinate real generalblock\n" );		
	else
		fprintf( fid, "%%%%MatrixMarket matrix coordinate real general\n" );
	fprintf( fid, "%d %d %d\n", A->nrows, A->ncols, A->nnz );
	if( A->nz )
	{
		if( !A->block )
		{
			for( row=1, i=0; i<A->nnz; i++ )
			{
				while( i>=A->rindx[row])
					row++;
				fprintf( fid, "%d %d %20g\n", row, A->cindx[i]+1,  A->nz[i] );
			}
		}
		else
		{
			double *nzp = A->nz;
			
			for( row=1, i=0; i<A->nnz; i++, nzp+=(BLOCK_SIZE*BLOCK_SIZE) )
			{
				while( i>=A->rindx[row])
					row++;
				fprintf( fid, "%d %d %20g %20g %20g %20g\n", row, A->cindx[i]+1,  nzp[0],  nzp[1],  nzp[2],  nzp[3] );
			}
		}
	}
	else // pattern matrix
	{
		for( row=1, i=0; i<A->nnz; i++ )
		{
			while( i>=A->rindx[row])
				row++;
			fprintf( fid, "%d %d 1\n", row, A->cindx[i]+1 );
		}
	}
	
	
	fclose( fid );
	
	return 1;
}

/*
 output a matrix in a format that is very easy to load into Matlab
 */
int mtx_CRS_output_matlab( Tmtx_CRS_ptr A, char *fname )
{
	FILE *fid;
	int i, row;
	
	if( (fid=fopen( fname, "w"))==NULL )
	{
		fprintf( stdout, "ERROR : Unable to open file %s for the output of matrix data\n", fname );
		return 0;
	}
	
	if( A->block )
		fprintf( fid, "%d %d %d 0 0 0\n1 0 0 0 0 0\n", A->nrows, A->ncols, A->nnz );
	else
		fprintf( fid, "%d %d %d\n0 0 0\n", A->nrows, A->ncols, A->nnz );
	if( A->nz )
	{
		if( !A->block )
		{
			for( row=1, i=0; i<A->nnz; i++ )
			{
				while( i>=A->rindx[row])
					row++;
				fprintf( fid, "%d %d %20g\n", row, A->cindx[i]+1,  A->nz[i] );
			}
		}
		else
		{
			double *nzp = A->nz;
			
			for( row=1, i=0; i<A->nnz; i++, nzp+=(BLOCK_SIZE*BLOCK_SIZE) )
			{
				while( i>=A->rindx[row])
					row++;
				fprintf( fid, "%d %d %20g %20g %20g %20g\n", row, A->cindx[i]+1,  nzp[0],  nzp[1],  nzp[2],  nzp[3] );
			}
		}
	}
	else // pattern matrix
	{
		for( row=1, i=0; i<A->nnz; i++ )
		{
			while( i>=A->rindx[row])
				row++;
			fprintf( fid, "%d %d 1\n", row, A->cindx[i]+1 );
		}
	}
	
	
	fclose( fid );
	
	return 1;
}

/*
 *		output a compiled mesh to disk
 */
int mesh_compiled_save( Tmesh_ptr mesh, char *fname )
{
	FILE *fid;
	int i, n_nodes, n_elements;
	
	/*
	 *		open the file for output
	 */
	if( (fid=fopen( fname, "wb" ))==NULL )
	{
		fprintf( stderr, "ERROR : mesh_compiled_save() : unable to open the file %s for output\n\n", fname );
		return 0;
	}
	
	/*
	 *		output the data
	 */
	
	n_nodes    = mesh->n_nodes;
	n_elements = mesh->n_elements;
	
	// header information
	fwrite( &n_nodes, sizeof(int), 1, fid );
	fwrite( &n_elements, sizeof(int), 1, fid );
	
	// nodes
	fwrite( mesh->node_x, sizeof(double), n_nodes, fid );
	fwrite( mesh->node_y, sizeof(double), n_nodes, fid );	
	fwrite( mesh->node_z, sizeof(double), n_nodes, fid );
	fwrite( mesh->node_bc, sizeof(int), n_nodes, fid );
	fwrite( mesh->node_num_elements, sizeof(int), n_nodes, fid );
	for( i=0; i<n_nodes; i++ )
	{
		fwrite( mesh->node_elements[i], sizeof(int), mesh->node_num_elements[i], fid );
	}
	
	// CRS structure
	fwrite( &mesh->A.nnz, sizeof(int), 1, fid );					// nnz
	fwrite( mesh->A.rindx, sizeof(int), mesh->A.nrows+1, fid ); // rindx
	fwrite( mesh->A.cindx, sizeof(int), mesh->A.nnz, fid );		// cindx
	
	// elements
	for( i=0; i<n_elements; i++ )
	{
		fwrite( &mesh->elements[i].n_nodes, sizeof(int), 1, fid );
		fwrite( &mesh->elements[i].n_edges, sizeof(int), 1, fid );
		fwrite( &mesh->elements[i].n_faces, sizeof(int), 1, fid );
		fwrite( &mesh->elements[i].tag,   sizeof(int), 1, fid );
		
		fwrite( mesh->elements[i].nodes,    sizeof(int), mesh->elements[i].n_nodes, fid );
		fwrite( mesh->elements[i].face_bcs, sizeof(int), mesh->elements[i].n_faces, fid );
	}

	/*
	 *		close the output file
	 */
	fclose( fid );
	
	return 1;	
}

/*
 *		load a compiled mesh from disk
 */
int mesh_compiled_read( Tmesh_ptr mesh, char *fname )
{
	FILE *fid;
	int i, n_nodes, n_elements, nnz;
	
	/*
	 *		open the file for input
	 */
	if( (fid=fopen( fname, "rb" ))==NULL )
	{
		fprintf( stderr, "ERROR : mesh_compiled_read() : unable to open the file %s for output\n\n", fname );
		return 0;
	}
	
	/*
	 *		output the data
	 */
	
	// header information
	fread( &n_nodes, sizeof(int), 1, fid );
	fread( &n_elements, sizeof(int), 1, fid );
	mesh->n_nodes = n_nodes;
	mesh->n_elements = n_elements;
	
	// nodes
	mesh->node_x = (double*)malloc( sizeof(double)*n_nodes );
	mesh->node_y = (double*)malloc( sizeof(double)*n_nodes );
	mesh->node_z = (double*)malloc( sizeof(double)*n_nodes );
	mesh->node_bc = (int*)malloc( sizeof(int)*n_nodes );
	mesh->node_num_elements = (int*)malloc( sizeof(int)*n_nodes );
	fread( mesh->node_x, sizeof(double), n_nodes, fid );
	fread( mesh->node_y, sizeof(double), n_nodes, fid );	
	fread( mesh->node_z, sizeof(double), n_nodes, fid );
	fread( mesh->node_bc, sizeof(int), n_nodes, fid );
	fread( mesh->node_num_elements, sizeof(int), n_nodes, fid );
	mesh->node_elements = (int**)malloc( sizeof(int*)*n_nodes );
	for( i=0; i<n_nodes; i++ )
	{
		mesh->node_elements[i] = malloc( sizeof(int)*mesh->node_num_elements[i] );
		fread( mesh->node_elements[i], sizeof(int), mesh->node_num_elements[i], fid );
	}
	
	// CRS structure
	fread( &nnz, sizeof(int), 1, fid );
	mesh->A.init = 0;
	mtx_CRS_init( &mesh->A, n_nodes, n_nodes, nnz, 0 );
	free( mesh->A.nz );
	fread( mesh->A.rindx, sizeof(int), n_nodes+1, fid ); 
	fread( mesh->A.cindx, sizeof(int), nnz,       fid );		
	
	// elements
	mesh->elements = (Telement_ptr)malloc( sizeof(Telement)*n_elements );
	for( i=0; i<n_elements; i++ )
	{
		fread( &mesh->elements[i].n_nodes, sizeof(int), 1, fid );
		fread( &mesh->elements[i].n_edges, sizeof(int), 1, fid );
		fread( &mesh->elements[i].n_faces, sizeof(int), 1, fid );
		fread( &mesh->elements[i].tag,   sizeof(int), 1, fid );
		
		mesh->elements[i].nodes    = (int*)malloc( sizeof(int)*mesh->elements[i].n_nodes );
		mesh->elements[i].face_bcs = (int*)malloc( sizeof(int)*mesh->elements[i].n_faces );
		fread( mesh->elements[i].nodes,    sizeof(int), mesh->elements[i].n_nodes, fid );
		fread( mesh->elements[i].face_bcs, sizeof(int), mesh->elements[i].n_faces, fid );
	}
	
	/*
	 *		close the output file
	 */
	fclose( fid );
	
	return 1;	
}

/*
 *		save a distributed matrix in CRS form to disk
 *		
 *		if there are np processors, then np+1 files are output.
 *
 *		the file stub_np.dat contains data on the matrix distribution
 *		the files stub_np_Pid.mtx contain the CRS format matrix for each Pid
 */
int mtx_CRS_dist_output( Tmtx_CRS_dist_ptr A, char *stub )
{
	FILE *fid;
	char fname[256];
	int success, success_, this_dom=A->This.this_proc, n_dom=A->This.n_proc, i;
	
	if( !A->init )
	{
		if( !this_dom )
			fprintf( stderr, "ERROR : mtx_CRS_dist_output() : matrix must be initialised for output\n" );
		return 0;
	}
	
	// output the header file
	if( !this_dom )
	{		
		sprintf( fname, "%s_%d.dat", stub, n_dom );
		
		fid = fopen( fname, "w" );
		if( !fid )
		{
			fprintf( stderr, "ERROR : mtx_CRS_dist_output() : could not open the file %s for output\n", fname );
			success = 0;
		}
		else
		{
			// output stub and number of domains
			fprintf( fid, "%s\n%d\n%d\n%d\n", stub, n_dom, A->ncols, A->mtx.block );
			// output vtxdist 
			for( i=0; i<=n_dom; i++ )
				fprintf( fid, "%d ", A->vtxdist[i] );
			fprintf( fid, "\n" );
			
			success = 1;
		}
		
		fclose( fid );
	}
	
	// everyone verify that output of header data on the root Pid was successful
	MPI_Bcast( &success, 1, MPI_INT, 0, A->This.comm );
	if( !success )
		return 0;
	
	// output the individual CRS matrices
	sprintf( fname, "%s_%d_%d.mtx", stub, n_dom, this_dom );
	success = mtx_CRS_output( &A->mtx, fname );
	
	// everyone verify that output of matrix data on each Pid was successful
	MPI_Allreduce( &success, &success_, 1, MPI_INT, MPI_LAND, A->This.comm );
	if( !success_ )
		return 0;
	
	return 1;
}

int mtx_CRS_dist_load( Tmtx_CRS_dist_ptr A, char *stub, TMPI_dat_ptr This )
{
	FILE *fid;
	char fname[256];
	int success_, success, this_dom=This->this_proc, n_dom=This->n_proc, i, block, *vtxdist, nrows, ncols;
	
	/*
	 *		scan the header file
	 */
	sprintf( fname, "%s_%d.dat", stub, n_dom );
	
	fid = fopen( fname, "r" );
	if( !fid )
	{
		if( !this_dom )
			fprintf( stderr, "ERROR : mtx_CRS_dist_load() : could not open the file %s for input\n", fname );
		return 0;
	}
	
	// read stub (ignore) and number of domains
	fscanf( fid, "%s %d %d %d", fname, &n_dom, &ncols, &block );
	
	// read vtxdist
	vtxdist = (int*)malloc( sizeof(int)*n_dom+1 );
	for( i=0; i<=n_dom; i++ )
		fscanf( fid, "%d", vtxdist + i );
	nrows = vtxdist[this_dom+1] - vtxdist[this_dom];
	
	fclose( fid );
	
	// this sets up the distributed CRS matrix with a local matrix of length 1
	// the local matrix is automatically updated in the call to mtx_CRS_load() below
	mtx_CRS_dist_init( A, nrows, ncols, 1, block, This );
	memcpy( A->vtxdist, vtxdist, sizeof(int)*(n_dom+1) );
	free( vtxdist );
	
	/*
	 *		read the individual CRS matrices 
	 */
	sprintf( fname, "%s_%d_%d.mtx", stub, n_dom, this_dom );
	success = mtx_CRS_load( fname, &A->mtx );
	
	// everyone verify that output of matrix data on each Pid was successful
	MPI_Allreduce( &success, &success_, 1, MPI_INT, MPI_LAND, A->This.comm );
	if( !success_ )
		return 0;
	
	// everyone verify that matrix data on each Pid is valid
	success = mtx_CRS_validate( &A->mtx );
	MPI_Allreduce( &success, &success_, 1, MPI_INT, MPI_LAND, A->This.comm );
	if( !success_ )
		return 0;
	
	return 1;
}

int domain_output( Tdomain_ptr dom, char *stub )
{
	char fname[256];
	int i, n_dom=dom->n_dom;
	FILE *fid;	
	
	// open the output file
	sprintf( fname, "%s_%d.dom", stub, n_dom );
	if( !(fid = fopen(fname,"w")) )
	{
		printf( "ERROR : domain_ouput() : unable to open file %s for output\n", fname );
		return 0;
	}
	
	// output the domain data
	fprintf( fid, "%d\n", n_dom );
	fprintf( fid, "%d\n", dom->n_node );
	for( i=0; i<n_dom+1; i++ )
		fprintf( fid, "%d ", dom->vtxdist[i] );
	fprintf( fid, "\n" );
	for( i=0; i<n_dom*n_dom; i++ )
		fprintf( fid, "%d ", dom->map[i] );
	fprintf( fid, "\n" );	
	for( i=0; i<n_dom; i++ )
		fprintf( fid, "%d ", dom->n_in[i] );
	fprintf( fid, "\n" );
	for( i=0; i<n_dom; i++ )
		fprintf( fid, "%d ", dom->n_bnd[i] );
	fprintf( fid, "\n" );
	
	// close the file
	fclose( fid );
	
	return 1;
}

int domain_load( Tdomain_ptr dom, char *stub, TMPI_dat_ptr This )
{
	char fname[256];
	int i, n_dom, n_node;
	FILE *fid;	
	
	// open the output file
	sprintf( fname, "%s_%d.dom", stub, This->n_proc );
	if( !(fid = fopen(fname,"r")) )
	{
		printf( "ERROR : domain_ouput() : unable to open file %s for output\n", fname );
		return 0;
	}
	
	// read the domain data
	fscanf( fid, "%d", &n_dom );	
	fscanf( fid, "%d", &n_node );
	
	domain_init( dom, This, n_node );
	
	for( i=0; i<n_dom+1; i++ )
		fscanf( fid, "%d", dom->vtxdist + i );
	for( i=0; i<n_dom*n_dom; i++ )
		fscanf( fid, "%d", dom->map + i );
	for( i=0; i<n_dom; i++ )
		fscanf( fid, "%d", dom->n_in + i );
	for( i=0; i<n_dom; i++ )
		fscanf( fid, "%d", dom->n_bnd + i );
	
	// close the file
	fclose( fid );
	
	return 1;
}

/*
 *		save a Jacobian matrix and relevant domain data so that a stand-alone
 *		program can load the matrix without any prior knowlege of the physical domain
 *		and manipulate the matrix with full domain awareness.
 */
int jacobian_output( Tmtx_CRS_dist_ptr A, Tdomain_ptr dom, char *stub )
{
	if( !domain_output( dom, stub ) )
		return 0;
	if( !mtx_CRS_dist_output( A, stub ) )
		return 0;
	
	return 1;
}

/*
 *		load a Jacobian matrix and relevant domain data.
 */
int jacobian_load( Tmtx_CRS_dist_ptr A, Tdomain_ptr dom, char *stub, TMPI_dat_ptr This )
{
	if( !domain_load( dom, stub, This ) )
		return 0;
	if( !mtx_CRS_dist_load( A, stub, This ) )
		return 0;
	
	return 1;
}

