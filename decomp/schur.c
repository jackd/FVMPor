/*
 *  schur.c
 *  
 *
 *  Created by Ben Cumming on 15/03/06.
 *
 */

#include "schur.h"

void GMRES_setup_schur( Tgmres_ptr g );
void mtx_CRS_dist_split_global( Tmtx_CRS_dist_ptr A, Tprecon_schur_ptr P );
void find_vtxdist( int n, TMPI_dat_ptr This, int block, int *vtxdist, int *vtxdist_b, int *vtxspace );
int unpacked_test_col( double *sdat, int n_bnd, int col ) ;

/******************************************************************************************
*       precon_schur_init()
*
*       perform basic initialisation of a schur  compliment preconditioner.
*       The preconditioner is for a matrix distributed over This and is for
*       the matrix A.
******************************************************************************************/
void precon_schur_init( Tprecon_schur_ptr P, Tmtx_CRS_dist_ptr A  )
{
        if( !P )
        {
                printf( "ERROR : precon_schur_init() : passed NULL pointer for preconditioner -- memory not allocated properly\n" );
                MPI_Finalize();
                exit(0);
        }
        
        // check that A has been initialised
        ASSERT_MSG( A->init, "precon_schur_init() : trying to initialise preconditioner for an uninitialised matrix" );
        
        if( P->init )
                precon_schur_free( P );
        
        // initialise the data structure
        P->domains.vtxdist = NULL;
        P->domains.map = NULL;
        P->domains.n_in = NULL;
        P->domains.n_bnd = NULL;
        P->forward.neighbours = P->backward.neighbours = NULL;
        P->forward.counts = P->backward.counts = NULL;
        P->forward.starts = P->backward.starts = NULL;
        P->forward.indx = P->backward.indx = NULL;
        P->forward.part = P->backward.part = NULL;
        P->forward.ppart = P->backward.ppart = NULL;
        P->GMRES_params.residuals = NULL;
        P->GMRES_params.errors = NULL;
        P->GMRES_params.j = NULL;
        P->GMRES_params.k = 0;
        P->GMRES_params.K = NULL;
        P->GMRES_params.diagnostic = NULL;      
        P->root    = 0;
        P->n_in    = 0;
        P->n_bnd   = 0;
        P->n_local = 0;
        P->n_neigh = 0;
        P->A.init  = 0;
        P->S.init  = 0;
        P->Slocal.init = 0;
        P->part_g  = (int*)malloc( sizeof(int)*A->nrows );
        P->q       = (int*)malloc( sizeof(int)*A->nrows );
        P->p       = (int*)malloc( sizeof(int)*A->nrows );
        BMPI_copy( &A->This, &P->This );
        P->MS.preconditioner = NULL;
        P->MB.preconditioner = NULL;
        P->MB.init = P->MS.init = 0;
        P->MS.parameters = NULL;
        P->MB.parameters = NULL;
        
        P->init    = 1;
}

/******************************************************************************************
*               mtx_CRS_dist_split()
*
*               take a distributed CRS matrix A and perform domain decomp
*               and split it into the schur form in the precondioner P
*******************************************************************************************/
void mtx_CRS_dist_split( Tmtx_CRS_dist_ptr A, Tprecon_schur_ptr P )
{
        int i, k, pos, this_dom, n_dom, tdom, sdom, r_pos, n_nodes, block, n_neigh, connect_dom, max_node_nnz;
        int index_pos, recv_nodes, is_boundary, nnz, n_in, n_bnd, n_node, nodes_max, nnz_max, posb, row;
        int *is_neigh, *back_counts, *p, *int_temp;
        int  *index_pack_send, *tag_pack_send, *nnz_pack_send, *index_pack_recv, *tag_pack_recv, *nnz_pack_recv;
        double *dbl_temp, *nz_pack_send, *nz_pack_recv;
        Tnode_ptr  node=NULL;
        Tnode_list in, bnd, nodes;
        TMPI_dat_ptr This;
        MPI_Request *request;
        MPI_Status *status;
        Tmtx_CRS_dist Ad;
        
        Ad.init = 0;
        
        /*
         *              Setup the diagnostic file for This thread
         *              
         *              This file contains diagnostic output describing the communication
         *              between processes/domains and the domain decompt method
         */
        
        This = &A->This;

        /*
         *              Setup initial data stuff
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
        mtx_CRS_dist_domdec( A, &P->forward, P->part_g, 1 );
        
        /* 
         *              Send and recieve nodes
         *
         *              The processors exchange nodes with one another so that at completion of This
         *              loop each processor has only the nodes corresponding to its domain.
         *              On each iteration of the loop P(i) will give nodes to P(i+k) and recieve
         *              nodes from P(i-k), allowing all processes to continue communicating 
         *              constantly.
         *
         *              The nodes are stored in node linked lists, not in CRS format, This allows
         *              easy sorting and manipulation of the nodes.
         */

        request = malloc( 20*sizeof(MPI_Request) );
        status  = malloc( 20*sizeof(MPI_Status) );
        for( i=0; i<20; i++ )
                status[i].MPI_ERROR = MPI_SUCCESS;
        nodes_max = 0;
        for( k=0; k<n_dom; k++ )
                if( nodes_max<P->forward.counts[k] )
                        nodes_max = P->forward.counts[k];
        nnz_max    = A->mtx.nnz;
        
        // initialise memory for send buffers
        tag_pack_send   = (int*)malloc( nodes_max*sizeof(int) );
        nnz_pack_send   = (int*)malloc( nodes_max*sizeof(int) );
        if( !block )
                nz_pack_send    = (double*)malloc( nnz_max*sizeof(double) );
        else
                nz_pack_send    = (double*)malloc( nnz_max*(sizeof(double) BLOCK_M_SHIFT) );
        index_pack_send = (int*)malloc( nnz_max*sizeof(int) );

        // initialise memory for receive buffers
        tag_pack_recv   = (int*)malloc( nodes_max*sizeof(int) );
        nnz_pack_recv   = (int*)malloc( nodes_max*sizeof(int) );
        if( !block )
                nz_pack_recv    = (double*)malloc( nnz_max*sizeof(double) );
        else
                nz_pack_recv    = (double*)malloc( nnz_max*(sizeof(double) BLOCK_M_SHIFT) );
        index_pack_recv = (int*)malloc( nnz_max*sizeof(int) );
        
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

                // the number of nodes to send to tdom
                n_nodes = P->forward.counts[tdom];      
                
                // Pack the data for sending to tdom
                nnz = 0;
                for( pos=0, i=P->forward.starts[tdom]; i<P->forward.starts[tdom+1]; pos++, i++ )
                {
                        index_pos = A->mtx.rindx[P->forward.indx[i]];
                        tag_pack_send[pos]   = A->vtxdist[this_dom] + P->forward.indx[i];
                        nnz_pack_send[pos]   = A->mtx.rindx[P->forward.indx[i]+1] - index_pos;

                        memcpy( index_pack_send + nnz, A->mtx.cindx + index_pos, nnz_pack_send[pos]*sizeof(int) );
                        if( !block )
                                memcpy( nz_pack_send + nnz,    A->mtx.nz + index_pos,    nnz_pack_send[pos]*sizeof(double) );
                        else
                                memcpy( nz_pack_send + (nnz BLOCK_M_SHIFT),    A->mtx.nz + (index_pos BLOCK_M_SHIFT),    nnz_pack_send[pos]*(sizeof(double) BLOCK_M_SHIFT) );                           
                                                
                        nnz+=nnz_pack_send[pos];
                }
                
                // send the number of nodes being passed
                MPI_Isend( &n_nodes, 1,      MPI_INT, tdom, TAG_NODES, This->comm, request+r_pos );
                r_pos++;
                
                // send packed data tdom
                if( n_nodes )
                {
                        // tags
                        MPI_Isend( tag_pack_send, n_nodes, MPI_INT, tdom, TAG_TAG, This->comm, request+r_pos );
                        r_pos++;
                        
                        // nnz
                        MPI_Isend( nnz_pack_send,   n_nodes, MPI_INT, tdom, TAG_NNZ, This->comm, request+r_pos );
                        r_pos++;
                        
                        // data
                        MPI_Isend( index_pack_send,  nnz,     MPI_INT, tdom, TAG_INDEX, This->comm, request+r_pos );
                        r_pos++;
                        
                        // data
                        if( !block )
                                MPI_Isend( nz_pack_send,     nnz,     MPI_DOUBLE, tdom, TAG_NZ, This->comm, request+r_pos );
                        else
                                MPI_Isend( nz_pack_send,     nnz BLOCK_M_SHIFT,     MPI_DOUBLE, tdom, TAG_NZ, This->comm, request+r_pos );
                                
                        r_pos++;
                }
                        
                /*
                 *              Receive data from the source Pid sdom
                 */
                
                // wait for the source to have sent its data to you. This could be acheived with some asynchronous
                // carry-on, but I want to minimise the complexity of this code
                MPI_Barrier( This->comm );
                
                // number of nodes
                MPI_Irecv( &recv_nodes, 1, MPI_INT, sdom, TAG_NODES , This->comm, request+r_pos );
                r_pos++;
                
                // check that we aren't going to run out of memory
                if( recv_nodes>nodes_max )
                {
                        nodes_max = recv_nodes;
                        tag_pack_recv = (int*)realloc( tag_pack_recv, nodes_max*sizeof(int) );
                        if( !block )
                                nnz_pack_recv = (int*)realloc( nnz_pack_recv, nodes_max*sizeof(int) );
                        else
                                nnz_pack_recv = (int*)realloc( nnz_pack_recv, nodes_max*(sizeof(int) BLOCK_M_SHIFT) );                          
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
                                nnz_max = nnz;
                                index_pack_recv =    (int*)realloc( index_pack_recv, nnz_max*sizeof(int) );
                                if( !A->mtx.block )
                                        nz_pack_recv    = (double*)realloc( nz_pack_recv,    nnz_max*sizeof(double) );
                                else
                                        nz_pack_recv    = (double*)realloc( nz_pack_recv,    nnz_max*(sizeof(double) BLOCK_M_SHIFT) );
                        }
                        
                        // indices
                        MPI_Irecv( index_pack_recv,  nnz, MPI_INT, sdom, TAG_INDEX, This->comm, request+r_pos );
                        r_pos++;
                        
                        // nz values
                        if( !block )
                                MPI_Irecv( nz_pack_recv,  nnz,   MPI_DOUBLE, sdom, TAG_NZ, This->comm, request+r_pos );
                        else
                                MPI_Irecv( nz_pack_recv,  nnz BLOCK_M_SHIFT,   MPI_DOUBLE, sdom, TAG_NZ, This->comm, request+r_pos );

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
                 *              Unpack data into node list
                 */
                for( i=0, nnz=0; i<recv_nodes; i++ )
                {
                        // allocate memory for the new node
                        if( !node_list_add( &nodes, nnz_pack_recv[i], tag_pack_recv[i] ) )
                                fprintf( stderr, "\tP%d : ERROR, unable to add node with tag %d\n", this_dom, tag_pack_recv[i] );

                        // copy over the indices and nz values
                        memcpy( nodes.opt->indx, index_pack_recv + nnz, sizeof(int)*nnz_pack_recv[i] );
                        if( !block )
                                memcpy( nodes.opt->dat, nz_pack_recv + nnz, sizeof(double)*nnz_pack_recv[i] );
                        else
                                memcpy( nodes.opt->dat, nz_pack_recv + (nnz BLOCK_M_SHIFT), (sizeof(double) BLOCK_M_SHIFT)*nnz_pack_recv[i] );
                        
                        nnz += nnz_pack_recv[i];
                }
        }
        
        max_node_nnz = 0;
        node = nodes.start;
        while( node!=NULL )
        {
                if( node->n>max_node_nnz )
                        max_node_nnz = node->n;
                node = node->next;
        }
        free( index_pack_send );
        free( nnz_pack_send );
        free( tag_pack_send );
        free( nz_pack_send );
        free( index_pack_recv );
        free( nnz_pack_recv );
        free( tag_pack_recv );
        free( nz_pack_recv );
        free( request );
        free( status  );
        
        /*
         *              Determine the internal and boundary nodes for This domain.
         *
         *              Internal and boundary nodes are sorted into the lists in and out
         *              respectively. The element is_neigh[i] records how many edges This
         *              domain shairs with domain i. Remember, the number of edges is NOT
         *              the number of nodes connected to. Think of is_neigh[i] as the nnz
         *              in the Ei for This domain.
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
                
        /*
         *              Processes send one-another information on who they are connected to
         *              
         *              This information is stored in map, where map[i*ndom:(i+1)*ndom-1] contains the is_neigh
         *              array for domain i.
         *
         *              The processors let everyone else know how many nodes they now have, allowing each
         *              process to build a new vtxdist for the new domain decomposition.
         */
        domain_init( &P->domains, This, n_node );
        
        // MPI command to distribute information
        MPI_Allgather( is_neigh, n_dom, MPI_INT, P->domains.map,     n_dom, MPI_INT, This->comm );
        
        // determine vtxdist for This distribution
        int_temp = (int *)malloc( sizeof(int)*n_dom );
        MPI_Allgather( &n_node,  1,     MPI_INT, int_temp, 1,     MPI_INT, This->comm );
        
        P->domains.vtxdist[0] = 0;
        for( i=1; i<=n_dom; i++ )
                P->domains.vtxdist[i] = P->domains.vtxdist[i-1]+int_temp[i-1];
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
                P->backward.starts[i] = P->backward.starts[i-1] + P->backward.counts[i-1];
        
        /* 
         *              calculate and store the global permutation vector
         */
        
        // allocate memory for temporary working arrays
        int_temp = (int *)malloc( sizeof(int)*n_dom );
        
        // determine how many nodes are stored on each processor
        for( i=0; i<n_dom; i++ )
                int_temp[i] = P->domains.vtxdist[i+1] - P->domains.vtxdist[i];
        
        // gather each domain's local ordering for the original node tags
        MPI_Allgatherv( P->backward.ppart, int_temp[this_dom], MPI_INT, P->p, int_temp, P->domains.vtxdist, MPI_INT, This->comm );
        
        // convert into global permutation
        for( i=0; i<A->nrows; i++ )
                P->q[P->p[i]] = i;

        // free temporary work arrays
        free( int_temp );
        
        /*
         *              Store the new reordered matrix in Ad
         */
        
        // determine the number of nonzeros in the local matrix
        nnz = 0;
        node = in.start;
        while( node!=NULL )
        {
                nnz += node->n;         
                node = node->next;
        }
        node = bnd.start;
        while( node!=NULL )
        {
                nnz += node->n;         
                node = node->next;
        }
        
        // initialise the matrix
        mtx_CRS_dist_init( &Ad, n_in+n_bnd, A->mtx.ncols, nnz, block, &A->This );
        
        // store matrix entries
        pos = posb = row = 0;
        Ad.mtx.rindx[0] = 0;
        int_temp = (int *)malloc( sizeof(int)*max_node_nnz );
        node = in.start;
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
                        dbl_temp = Ad.mtx.nz + (pos BLOCK_M_SHIFT); //(double *)malloc( sizeof(double)*(node->n BLOCK_M_SHIFT) );
                        permuteB( node->dat, dbl_temp, int_temp, node->n, 1 );
                }
                else
                {
                        dbl_temp = Ad.mtx.nz + pos; //(double *)malloc( sizeof(double)*node->n );
                        permute( node->dat, dbl_temp, int_temp, node->n, 1 );
                }               
                
                // copy over the column indices
                memcpy( Ad.mtx.cindx + pos, node->indx, node->n*sizeof(int) );
                
                pos += node->n;
                posb += node->n BLOCK_M_SHIFT;
                row++;
                Ad.mtx.rindx[row] = pos;
                node = node->next;
        }
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
                        dbl_temp = Ad.mtx.nz + (pos BLOCK_M_SHIFT);
                        permuteB( node->dat, dbl_temp, int_temp, node->n, 1 );
                }
                else
                {
                        dbl_temp = Ad.mtx.nz + pos;
                        permute( node->dat, dbl_temp, int_temp, node->n, 1 );
                }               
                
                // copy over the column indices
                memcpy( Ad.mtx.cindx + pos, node->indx, node->n*sizeof(int) );
                
                pos += node->n;
                posb += node->n BLOCK_M_SHIFT;
                row++;
                Ad.mtx.rindx[row] = pos;
                node = node->next;
        }
        free( int_temp );
        
        mtx_CRS_dist_split_global( &Ad, P );    
        
        // free up stuff
        node_list_free( &in );
        node_list_free( &bnd );
        mtx_CRS_dist_free( &Ad );


// BUGFIX 2006
// -----------
    free( is_neigh );
// -----------

}

/******************************************************************************************
*               mtx_CRS_dist_split_global()
*
*               take a distributed CRS matrix A and split it into the schur form in 
*               the precondioner P. The matrix A is already in the decomposed form.
*
*               P has been initialised and A is initialised and valid.
*******************************************************************************************/
void mtx_CRS_dist_split_global( Tmtx_CRS_dist_ptr A, Tprecon_schur_ptr P )
{
        TMPI_dat_ptr This;
        int n_dom, this_dom, block, n_in, n_bnd, n_neigh, dom, dom_start, this_dom_pos;
        int i, j, k, pos, jj, spot, row_nnz;
        int Ennz, Bnnz, Fnnz, Cnnz, offset, row_start;
        int *in_split, *cx, *neighbours, *Eijnnz, **bnd_split;
        double *nz;

        // initialise variables
        This = &A->This;
        block = A->mtx.block;
        n_dom = This->n_proc;
        this_dom = This->this_proc;
        

        /*
         *              Store the split of A
         *
         *              Done in two steps :
         *                      1.  all indices in nodes are changed to new order and sorted.
         *                              during This the number of nonzero entries in each of the split
         *                              matrices is calculated.
         *                      2.  memory is allocated for the split matrices and the node data is copied
         *                              into the splits.
         *
         */
        
        /* 
         *              do the interior nodes first, these contribute to the matrices B and E
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
        
        // allocate the split matrix
        mtx_CRS_split_init( &P->A, n_in+n_bnd, n_in, n_bnd, n_dom, this_dom, n_neigh );
        
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
                        {
                                P->A.E.cindx[i] -= offset;

                        }
                        if( block )
                                memcpy( P->A.E.nz + (P->A.E.rindx[j] BLOCK_M_SHIFT), nz + (in_split[j] BLOCK_M_SHIFT), (k BLOCK_M_SHIFT)*sizeof(double) );
                        else
                                memcpy( P->A.E.nz + P->A.E.rindx[j], nz + in_split[j], k*sizeof(double)  );
                }
        }

        // free up working arrays
        free( in_split );
        
        /* 
         *              do the exterior nodes, these contribute to the matrices F, C and the Eij
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
        for( i=0; i<n_neigh; i++ )
        {
                dom = neighbours[i];
                if( dom!=this_dom )
                {
                        mtx_CRS_init( P->A.Eij+i, n_bnd,  P->domains.n_bnd[dom], Eijnnz[i], block );
                        P->A.Eij[i].rindx[0] = 0;
                }
        }
        
        // now cut-up and store the nodes in the matrices
        for( j=0; j<n_bnd; j++ )
        {               
                jj = j + n_in;
                
                /* 
                 *      store the Eij rows for This node
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
        // save the list of neighbours
        
        memcpy( P->A.neighbours, neighbours, sizeof(int)*n_neigh );
        P->n_bnd = n_bnd;
        P->n_neigh = n_neigh;
        P->n_in = n_in;
        P->n_local = n_in + n_bnd;
        
        // free up working arrays
        free( in_split );
        free( neighbours );
        free( Eijnnz );
        for( k=0; k<n_bnd; k++ )
                free( bnd_split[k] );
        free( bnd_split );
}

/******************************************************************************************
*       schur_form_local()
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
         *              setup parameters
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
         *              Form CCS versions of E and C
         */
        
        mtx_CRS_to_CCS( C, &C_CCS );
        mtx_CRS_to_CCS( E, &E_CCS );
        
        /*
         *              create the columns of S one-at-a-time and store in place
         */
        S->cindx[0] = 0;
        sp_pos = 0;
        for( col=0; col<n_bnd; col++ )
        {                       
                // check if we need to allocate more memory in S
                while( (S->nnz - sp_pos)<n_bnd )
                {
                        // allocate space for 1.2 times more nz entries per column than are
                        // currently being used
                        S->nnz = ceil( (double)sp_pos/(double)col*1.2 )*n_bnd;
                        
                        S->rindx = (int *)   realloc( S->rindx, sizeof(int)*S->nnz );
                        S->nz    = (double *)realloc( S->nz,    sizeof(double)*S->nnz );
                        
                }
                
                // form the dense column of E
                for( i=0; i<n_in; i++ )
                        x.dat[i] = 0.;
                for( pos=E_CCS.cindx[col]; pos<E_CCS.cindx[col+1]; pos++ )
                        x.dat[E_CCS.rindx[pos]] = E->nz[pos];
                
                // apply preconditioner to it, storing in y
                precon_apply( P, &x, &y );
                
                // multiply F on LHS  of preconditioned column
                // s = -F*y;
                s = S->nz + S->cindx[col];
                mtx_CRS_gemv( F, y.dat, s, -1., 0., 'n' );
                        
                // adjust with the relevant values from C
                for( pos=C_CCS.cindx[col]; pos<C_CCS.cindx[col+1]; pos++ )
                        s[C_CCS.rindx[pos]] += C->nz[pos];
                
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
         *              clean up
         */
        vec_dist_free( &x );
        vec_dist_free( &y );
        mtx_CCS_free( &C_CCS );
        mtx_CCS_free( &E_CCS );
}

/******************************************************************************************
*       schur_form_local_block()
*       
*   create a local schur compliment matrix. P is the preconditioner of the matrix B.
*       params is the parameters for the schur compliment preconditioner
******************************************************************************************/
void schur_form_local_block( Tprecon_ptr P, Tmtx_CRS_ptr E, Tmtx_CRS_ptr F, Tmtx_CRS_ptr C, Tmtx_CCS_ptr S, Tprecon_schur_params_ptr params )
{
        int vtxdist[2];
        Tmtx_CCS E_CCS, C_CCS;
        int n_in, n_bnd, colfill;
        double droptol;
        int i, col, pos, sp_pos, nz_count;
        double *s, *edat, *ydat, *sdat, *copy_from, *copy_to, *drop_temp;
        Tvec_dist x, y;

        /*
         *              setup parameters
         */
        
        y.init = x.init = E_CCS.init = C_CCS.init = 0;
        
        // find the number of internal and boundary nodes
        n_in = E->nrows;
        n_bnd = E->ncols;
        
        // make sure that setup lfill parameter isn't too large
        droptol = params->droptol;
        colfill = (params->lfill<n_bnd) ? params->lfill : n_bnd;

        // allocate memory
        vtxdist[0] = 0;
        vtxdist[1] = n_in;
        vec_dist_init( &x, &P->This, n_in, E->block, vtxdist );
        vec_dist_init( &y, &P->This, n_in, E->block, vtxdist );
        mtx_CCS_init( S, n_bnd, n_bnd, colfill*n_bnd + n_bnd, E->block );
        edat = (double*)calloc( sizeof(double), (n_in BLOCK_M_SHIFT) );
        ydat = (double*)calloc( sizeof(double), (n_in BLOCK_M_SHIFT) );
        sdat = (double*)calloc( sizeof(double), (n_bnd BLOCK_M_SHIFT) );
        drop_temp = (double*)calloc( sizeof(double), n_bnd );


        /*
         *              Form CCS versions of E and C
         */

        mtx_CRS_to_CCS( C, &C_CCS );
        mtx_CRS_to_CCS( E, &E_CCS );

        /*
         *              create the columns of S one-at-a-time and store in place
         */
        S->cindx[0] = 0;
        for( col=0; col<n_bnd; col++ )
        {
                // form the dense columns of E
                mtx_CCS_column_unpack_block( &E_CCS, edat, col );

                // apply preconditioner to it, storing in y
                for( i=0; i<BLOCK_SIZE; i++ )
                {
                        memcpy( x.dat, edat + i*n_in*BLOCK_SIZE, sizeof(double)*(n_in BLOCK_V_SHIFT) );
                        precon_apply( P, &x, &y );
                        memcpy( ydat + i*n_in*BLOCK_SIZE, y.dat, sizeof(double)*(n_in BLOCK_V_SHIFT) );

                }

                // multiply F on LHS  of preconditioned column
                // s = -F*y;
                for( i=0; i<BLOCK_SIZE; i++ )
                {
                        s = sdat + i*n_bnd*BLOCK_SIZE;
                        mtx_CRS_gemv( F, ydat + i*n_in*BLOCK_SIZE, s, -1., 0., 'n' );
                }

                // adjust with the relevant values from s = C + s =     C - F*inv(B)*E
                mtx_CCS_column_add_unpacked_block( &C_CCS, sdat, col, 1., 1. );

                // apply dropping to the column
                nz_count = vec_drop_block( sdat, S->rindx+S->cindx[col], n_bnd, colfill, col, droptol, drop_temp );

                // store the new (block) column of the schur comliment
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
         *              clean up
         */
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
*       schur_form_global()
*       
*   create the local chunk of the global schur compliment matrix.
*
*   P is the schur compliment preconditioner to which the schur chunk is to be added.
*
*   This is a bit fiddly, but there isn't really very much that can be done about it.
******************************************************************************************/
void schur_form_global( Tprecon_schur_ptr P )
{
        int nnz, n_neigh, Sdim, n_bnd, this_dom, i, dom_i, pos;
        int dstart, len, n_dom, dom, row, block;
        Tmtx_CRS_ptr Ep;
        int *cindx, *Ecindx, *neighbours;
        double *nz;
        
        /*
         *              setup
         */
                
        this_dom = P->This.this_proc;
        n_dom = P->This.n_proc;
        
        // block?
        block = P->Slocal.block;
        
        // how many neighbouring domains does This domain have?
        n_neigh = P->n_neigh;
        
        // how many boundary nodes in This domain
        n_bnd = P->n_bnd;
        
        
        // determine the total number of boundary nodes in the global domain
        Sdim = 0;
        for( i=0; i<n_dom; i++ )
                Sdim += P->domains.n_bnd[i];
        
        // neighbours points to the list of neighbours of This domain
        neighbours = P->A.neighbours;
        
        /*
         *              augment the matrices
         */
        
        // find total number of nz elements
        nnz = P->Slocal.nnz;
        for( i=0; i<n_neigh; i++ )
                if( neighbours[i]!=this_dom )
                        nnz += P->A.Eij[i].nnz;
        
        // initialise the global schur
        mtx_CRS_dist_init( &P->S, n_bnd, Sdim, nnz, P->A.B.block, &P->This );
        
        // create the schur, one row at a time
        cindx = P->S.mtx.cindx;
        nz    = P->S.mtx.nz;
        P->S.mtx.rindx[0] = 0;
        
        for( row=0, pos=0; row<n_bnd; row++ )
        {
                dom_i = 0;
                Ep = P->A.Eij;
                
                
                // add parts from domains 0..this_dom-1
                while( (dom=neighbours[dom_i])<this_dom )
                {
                        Ecindx = Ep->cindx + Ep->rindx[row];
                        dstart = P->S.vtxdist[ dom ];
                        len = Ep->rindx[row+1] - Ep->rindx[row];
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
}

/******************************************************************************************
*       precon_schur()
******************************************************************************************/
int precon_schur( Tmtx_CRS_dist_ptr A, Tprecon_ptr preconditioner, int level )
{
        Tmtx_CCS S_CCS_tmp;
        Tmtx_CRS_dist B_dist;
        Tprecon_schur_ptr P;
        Tprecon_schur_params_ptr params;
        
        S_CCS_tmp.init = B_dist.init = 0;

        // check that the matrix is valid
        ASSERT_MPI( mtx_CRS_validate( &A->mtx ), "precon_schur() : the matrix to precondition is invalid, something has gone wrong in generating it", A->This.comm );
        
        /*
         *              Initialise the preconditioner
         */
        P = preconditioner->preconditioner;
        params = preconditioner->parameters;
        P->initial_decomp = 0;
        P->level = level;
        
        /*
         *              Form the preconditioner for the schur compliment if we are on level 0
         */
        if( !level )
        {       
                // store the distributed matrix to the preconditioner
                mtx_CRS_dist_copy( A, &P->S );
                
                // initialise and form the preconditioner
                precon_init( &P->MS, A, params->precon_type_S );
                if( !precon_params_load( P->MS.parameters, params->precon_fname_S, params->precon_type_S ) )
                {
                        sERROR( sprintf( _errmsg, "precon_schur() : unable to open file %s for level 0 Schur preconditioner parameters", params->precon_fname_S ) );
                        return 0;
                }               
                return precon( &P->MS, A, NULL );
        }
        
        /*
         *              Otherwise we need to perform another recursion
         */
        else
        {       
                // perform domain decomposition and distribute the matrix amongst the nodes
                mtx_CRS_dist_split( A, P );
                
                // form preconditioner for the local B matrix
                mtx_CRS_distribute( &P->A.B, &B_dist, P->This.sub_comm, 0 );
                precon_init( &P->MB, &B_dist, params->precon_type_B );
                if( !precon_params_load( P->MB.parameters, params->precon_fname_B, params->precon_type_B ) )
                {
                        sERROR( sprintf( _errmsg, "precon_schur() : unable to open file %s for local Bi preconditioner parameters", params->precon_fname_B ) );
                        return 0;
                }
                precon( &P->MB, &B_dist, NULL );
                mtx_CRS_dist_free( &B_dist );
                
                // form the schur compliment
                if( A->mtx.block )
                        schur_form_local_block( &P->MB, &P->A.E, &P->A.F, &P->A.C, &S_CCS_tmp, params );
                else
                        schur_form_local( &P->MB, &P->A.E, &P->A.F, &P->A.C, &S_CCS_tmp );
                
                // convert the local schur compliment to CRS fomrat
                P->Slocal.init = 0;
                mtx_CCS_to_CRS( &S_CCS_tmp, &P->Slocal );
                mtx_CCS_free( &S_CCS_tmp );
                
                // form the global schur compliment
                schur_form_global( P );

                // precondition the global schur compliment
                precon_init( &P->MS, &P->S, PRECON_SCHUR );
                precon_schur_params_copy( params, P->MS.parameters );
                return precon_schur( &P->S, &P->MS, level-1 );
        }
}

/******************************************************************************************
*       precon_schur_global()
*
*   form a schur compliment preconditioner in P for the matrix A where A has already been
*       decomposed and distributed. Distribution data is in dom
******************************************************************************************/
int precon_schur_global( Tmtx_CRS_dist_ptr A, Tprecon_ptr preconditioner, int level, Tdomain *dom )
{
        Tmtx_CCS S_CCS_tmp;
        Tmtx_CRS_dist B_dist;
        Tprecon_schur_ptr P;
        Tprecon_schur_params_ptr params;
        
        S_CCS_tmp.init = B_dist.init = 0;
        
        /*
         *              Initialise the preconditioner
         */
        P = preconditioner->preconditioner;
        params = preconditioner->parameters;
        P->level = level;
        P->initial_decomp = 1;
        domain_copy( dom, &P->domains );

        // Form the preconditioner for the schur compliment if we are on level 0
        if( !level )
        {               
                // store the distributed matrix to the preconditioner
                mtx_CRS_dist_copy( A, &P->S );
                
                // initialise and find the preconditioner
                precon_init( &P->MS, A, params->precon_type_S );
                if( !precon_params_load( P->MS.parameters, params->precon_fname_S, params->precon_type_S ) )
                {
                        sERROR( sprintf( _errmsg, "precon_schur_global() : unable to open file %s for level 0 Schur preconditioner parameters", params->precon_fname_S ) );
                        return 0;
                }
                return precon( &P->MS, A, NULL );
        }
        
        // Otherwise we need to perform another recursion
        else
        {
                //  Form the local split matrix
                mtx_CRS_dist_split_global( A, P );
                // form preconditioner for the local B matrix
                mtx_CRS_distribute( &P->A.B, &B_dist, P->This.sub_comm, 0 );
                precon_init( &P->MB, &B_dist, params->precon_type_B );
                //precon_params_copy( params->B_params, P->MB.parameters, params->precon_type_B  );
                if( !precon_params_load( P->MB.parameters, params->precon_fname_B, params->precon_type_B ) )
                {
                        sERROR( sprintf( _errmsg, "precon_schur_global() : unable to open file %s for local Bi preconditioner parameters", params->precon_fname_B ) );
                        return 0;
                }
                precon( &P->MB, &B_dist, NULL );
                mtx_CRS_dist_free( &B_dist );

                // form the schur compliment
                if( A->mtx.block )
                        schur_form_local_block( &P->MB, &P->A.E, &P->A.F, &P->A.C, &S_CCS_tmp, params );
                else
                        schur_form_local( &P->MB, &P->A.E, &P->A.F, &P->A.C, &S_CCS_tmp );
                                                
                // convert the local schur compliment to CRS fomrat
                P->Slocal.init = 0;
                mtx_CCS_to_CRS( &S_CCS_tmp, &P->Slocal );
                mtx_CCS_free( &S_CCS_tmp );
                                
                // form the global schur compliment
                schur_form_global( P );

                // precondition the global schur compliment
                precon_init( &P->MS, &P->S, PRECON_SCHUR );
                precon_schur_params_copy( params, P->MS.parameters );
                
                return precon_schur( &P->S, &P->MS, level-1 );
        }
}

/*************************************************************
 *              precon_schur_apply()
 *************************************************************/
void precon_schur_apply( Tprecon_ptr preconditioner, Tvec_dist_ptr x, Tvec_dist_ptr y )
{
        Tprecon_schur_ptr P;
        Tprecon_schur_params_ptr params;
        Tvec_dist xl, xL, yB, v, PxL, yL;
        int i, j, block, thisdom;
        double *tmp_double;
        char pre[100];
        int *vtxdist, *vtxdist_b, *counts;
        
        P = preconditioner->preconditioner;
        params = preconditioner->parameters;
        
        pre[0] = '\0';
        for( i=params->nlevels; i>P->level; i-- )
                strcat( pre, "\t" );
        thisdom = x->This.this_proc;
        block = P->Slocal.block;
        xl.init = xL.init = yB.init = v.init = PxL.init = yL.init = 0;
        
        /***************************************************************************************************
         *              if we are at the bottom level then solve the low level GMRES problem
         ***************************************************************************************************/   
        
        if( !P->level )
        {               
                if( params->GMRES )
                {
                        int success, success_;
                        
                        // load the GMRES parameters
                        if( !(success=GMRES_params_load( &P->GMRES_params, params->GMRES_fname )) )
                                printf( "P%d ERROR : precon_schur_global() : unable to open file %s for schur parms\n", thisdom, params->GMRES_fname );
                        
                        MPI_Allreduce( &success, &success_, 1, MPI_INT, MPI_LAND, P->This.comm );
                        if( !success_ )
                        {
                                if( success )
                                {
                                        printf( "P%d no error : precon_schur_global()\n", thisdom );
                                }
                                MPI_Finalize();
                                exit(1);
                        }
                        
                        // solve with GMRES : y = inv(S)*x
                        vec_dist_init_vec( y, x );
                        gmres( &P->S, x, y, &P->GMRES_params, &P->MS, 0 );
                }
                else
                {
                        precon_apply( &P->MS, x, y );
                }

                return;
        }
        
        /***************************************************************************************************
         *              Apply the forward preconditioning step
         ***************************************************************************************************/ 
        
        if( !P->initial_decomp )
        {
                double *x_global, *y_global, *y_local;
                int nlocal;
                
                // create the counts and vtxdist vectors for distribution
                counts   = (int*)malloc( sizeof(int)*(P->This.n_proc)   );
                vtxdist = (int*)malloc( sizeof(int)*(P->This.n_proc+1) );
                vtxdist_b = (int*)malloc( sizeof(int)*(P->This.n_proc+1) );
                find_vtxdist( P->n_local, &P->This, x->block, vtxdist, vtxdist_b, counts );
                
                // gather the x vector to the local CPU
                x_global = (double *)malloc( x->vtxdist_b[x->This.n_proc] * sizeof(double) );
                vec_dist_gather_all( x, x_global );
                                
                // now form our local part of the distributed vector
                j = vtxdist[ x->This.this_proc ];
                if( !x->block )
                {
                        tmp_double = (double *)malloc( P->n_local*sizeof(double) );
                        for( i=0; i<P->n_local ; i++, j++ )
                                tmp_double[i] = x_global[P->p[j]];  
                }
                else
                {
                        int ib;
                        
                        tmp_double = (double *)malloc( P->n_local*(sizeof(double) BLOCK_V_SHIFT) );
                        for( i=0, ib=0; i<P->n_local ; i++, j++, ib+=BLOCK_SIZE )
                                memcpy( tmp_double+ib, x_global+(P->p[j] BLOCK_V_SHIFT), BLOCK_SIZE*sizeof(double) );
                }

                // distribute the vector over the local comm
                vec_dist_scatter( &xL, tmp_double, x->This.sub_comm, x->block, P->n_in, 0 );

                // apply the local preconditioner to the vector entries corresponding the the local nodes
                precon_apply( &P->MB, &xL, &PxL );
        
                // allocate memory for the vector to pass to the next level
                vec_dist_init( &v, &P->This, P->n_bnd, x->block, NULL );

                if( block )
                        memcpy( v.dat, tmp_double + (P->n_in BLOCK_V_SHIFT), (P->n_bnd BLOCK_V_SHIFT)*sizeof(double) ); 
                else
                        memcpy( v.dat, tmp_double + P->n_in, P->n_bnd*sizeof(double) );
        
                // free memory used in all of this
                free( tmp_double );
                free( x_global );

                // v = v - F*PxL
                mtx_CRS_gemv( &P->A.F, PxL.dat, v.dat, -1, 1, 'n' );
        
                // Pass the forward preconditioned boundary data onto the next level of the schur preconditioner
                // yi = P( gi - Fi*MBi*fi ) stored in yB
                precon_schur_apply( &P->MS, &v, &yB );
        
                /*
                 *              Perform the backward preconditioning step 
                 */

                // apply backward preconditioner
        
                // fi - Ei*yi stored in xL
                mtx_CRS_gemv( &P->A.E, yB.dat, xL.dat, -1, 1, 'n' );
        
                // xi = MBi * (fi - Ei*yi)  stored in yL
                precon_apply( &P->MB, &xL, &yL );
                
                nlocal = yL.n + yB.n;
                
                // initialise the vector to receive in
                vec_dist_init_vec( y, x );
                
                // create the send buffer
                y_local = (double *)malloc( nlocal * sizeof(double) );
                memcpy( y_local,        yL.dat, yL.n*sizeof(double) );
                memcpy( y_local + yL.n, yB.dat, yB.n*sizeof(double) );
                
                // gather the global y vector to the local CPU
                y_global = (double *)malloc( y->vtxdist_b[y->This.n_proc] * sizeof(double) );
                MPI_Allgatherv( y_local, counts[thisdom], MPI_DOUBLE, y_global, counts, vtxdist_b, MPI_DOUBLE, P->This.comm ); // y_global = global y vector

                // form our local part of the distributed vector
                j = x->vtxdist[thisdom];
                tmp_double = y->dat;
                if( !y->block )
                {
                        for( i=0; i<y->n; i++, j++ )
                                tmp_double[i] = y_global[P->q[j]];
                }
                else
                {
                        int ib, nb;
                        nb = y->vtxdist[thisdom+1]-y->vtxdist[thisdom];
                        
                        for( i=0, ib=0; i<nb ; i++, j++, ib+=BLOCK_SIZE )
                                memcpy( tmp_double+ib, y_global+(P->q[j] BLOCK_V_SHIFT), BLOCK_SIZE*sizeof(double) );
                }

                // free memory
                vec_dist_free( &xl );
                vec_dist_free( &yB );
                vec_dist_free( &xL );
                vec_dist_free( &yL );
                vec_dist_free( &PxL );
                vec_dist_free( &v );
                free( counts );
                free( vtxdist );
                free( vtxdist_b );
                free( y_local );                
                free( y_global );
        }
        else
        {
                vec_dist_scatter( &xL, x->dat, x->This.sub_comm, x->block, P->n_in, 0 );

                // apply the local preconditioner to the vector entries corresponding the the local nodes
                precon_apply( &P->MB, &xL, &PxL );
        
                // allocate memory for the vector to pass to the next level
                vec_dist_init( &v, &P->This, P->n_bnd, x->block, NULL );
                
                if( block )
                        memcpy( v.dat, x->dat + (P->n_in BLOCK_V_SHIFT), (P->n_bnd BLOCK_V_SHIFT)*sizeof(double) );     
                else
                        memcpy( v.dat, x->dat + P->n_in, P->n_bnd*sizeof(double) );

                // v = v - F*PxL
                mtx_CRS_gemv( &P->A.F, PxL.dat, v.dat, -1, 1, 'n' );
        
                // Pass the forward preconditioned boundary data onto the next level of the schur preconditioner
                // yi = P( gi - Fi*MBi*fi ) stored in yB
                precon_schur_apply( &P->MS, &v, &yB );
        
                /*
                 *              Perform the backward preconditioning step 
                 */

                // apply backward preconditioner
        
                // fi - Ei*yi stored in xL
                mtx_CRS_gemv( &P->A.E, yB.dat, xL.dat, -1, 1, 'n' );
        
                // xi = MBi * (fi - Ei*yi)  stored in yL
                precon_apply( &P->MB, &xL, &yL );

                vec_dist_init_vec( y, x );
                
                if( block )
                {
                        memcpy( y->dat,                           yL.dat, (P->n_in  BLOCK_V_SHIFT)*sizeof(double) );
                        memcpy( y->dat + (P->n_in BLOCK_V_SHIFT), yB.dat, (P->n_bnd BLOCK_V_SHIFT)*sizeof(double) );
                }
                else
                {
                        memcpy( y->dat,           yL.dat, P->n_in*sizeof(double) );
                        memcpy( y->dat + P->n_in, yB.dat, P->n_bnd*sizeof(double) );
                }
                
                // free memory
                vec_dist_free( &xl );
                vec_dist_free( &yB );
                vec_dist_free( &xL );
                vec_dist_free( &yL );
                vec_dist_free( &PxL );
                vec_dist_free( &v );
        }       
}

void GMRES_setup_schur( Tgmres_ptr g )
{
        g->dim_krylov = 20;
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

int precon_schur_params_load( Tprecon_schur_params_ptr p, char *fname )
{
        FILE * fid;
        
        // open parameter file for input
        if( !(fid=fopen(fname,"r")) )
        {
                printf( "precon_schur_params_load() : unable to read parameters from file %s\n\n", fname );
                return 0;
        }
        
        /***********************************************
         *      read the parameters from the input file
         ***********************************************/
        
        //printf( "opened file %s\n", fname );
        if( !fgetvar( fid, "%d", &p->nlevels ) )                return 0;
        if( !fgetvar( fid, "%d", &p->initial_decomp ) ) return 0;
        if( !fgetvar( fid, "%d", &p->lfill ) )                  return 0;
        if( !fgetvar( fid, "%lf", &p->droptol ) )               return 0;
        if( !fgetvar( fid, "%d", &p->precon_type_B ) )  return 0;
        if( !fgetvar( fid, "%s", &p->precon_fname_B ) ) return 0;
        if( !fgetvar( fid, "%d", &p->precon_type_S ) )  return 0;
        if( !fgetvar( fid, "%s", &p->precon_fname_S ) ) return 0;
        if( !fgetvar( fid, "%d", &p->GMRES ) )                  return 0;
        if( !fgetvar( fid, "%s", &p->GMRES_fname ) )    return 0;

        // close input file and return success
        fclose(fid);
        return 1;
}

void precon_schur_free( Tprecon_schur_ptr P )
{

// -----------
// BUGFIX 2006

// Some memory must still be freed even if !P->S.init

/*
        // check that we have been asked to free an initialised preconditioner
        if( !P->init || !P->S.init ) {

                return;

    }
*/

        // check that we have been asked to free an initialised preconditioner
        if( !P->init) return;

        free( P->part_g );
        free( P->q );
        free( P->p );

        if(!P->S.init ) return;

// -----------

        // are we on the last level?
        if( !P->level )
        {               
                GMRES_params_free( &P->GMRES_params );
                precon_free( &P->MS );
                BMPI_free( &P->This );
                mtx_CRS_dist_free( &P->S );
                return;
        }

        // otherwise cleanup as usual
        domain_free( &P->domains );
        distribution_free( &P->forward );
        distribution_free( &P->backward );

// -----------
// BUGFIX 2006

// These must be freed in all cases, so this code is moved to earlier in the
// function.
/*
        if( P->part_g )
                free( P->part_g );
        if( P->q )
                free( P->q );
        if( P->p )
                free( P->p );
*/
// -----------

        mtx_CRS_split_free( &P->A );
        mtx_CRS_dist_free( &P->S );
        mtx_CRS_free( &P->Slocal );
        precon_free( &P->MB );
        if( P->MS.preconditioner )
                precon_free( &P->MS );
        BMPI_free( &P->This );
        
        P->init = 0;
}

// BUGFIX 2006
// -----------

// This function is not called anywhere, so rather than debug it like
// precon_schur_free, it is simply not defined.

/*
void precon_schur_clear( Tprecon_schur_ptr P )
{
        // check that we have been asked to free an initialised preconditioner
        if( !P->init )
                return;
        
        // are we on the last level?
        if( !P->level )
        {
                GMRES_params_free( &P->GMRES_params );
                precon_free( &P->MS );
                BMPI_free( &P->This );
                return;
        }
        
        // otherwise cleanup as usual
        //domain_free( &P->domains );
        distribution_free( &P->forward );
        distribution_free( &P->backward );
        
        if( P->part_g )
                free( P->part_g );
        if( P->q )
                free( P->q );
        if( P->p )
                free( P->p );

        mtx_CRS_split_free( &P->A );
        mtx_CRS_dist_free( &P->S );
        mtx_CRS_free( &P->Slocal );
        precon_free( &P->MB );
        precon_free( &P->MS );
        //BMPI_free( &P->This );
}

*/

// -----------

void precon_schur_params_copy( Tprecon_schur_params_ptr from, Tprecon_schur_params_ptr to )
{
        // check to ensure that valid arguements have been passed
        if( !from || !to )
        {
                printf( "ERROR : precon_schur_params_copy() : one of the arguments (to/from) is NULL\n" );
                MPI_Finalize();
                exit(1);
        }
        
        to->initial_decomp = from->initial_decomp;
        to->nlevels = from->nlevels;
        to->precon_type_B = from->precon_type_B;        
        to->precon_type_S = from->precon_type_S;
        to->GMRES = from->GMRES;
        to->lfill = from->lfill;
        to->droptol = from->droptol;
        sprintf( to->precon_fname_B, "%s", from->precon_fname_B );
        sprintf( to->precon_fname_S, "%s", from->precon_fname_S );
        sprintf( to->GMRES_fname, "%s", from->GMRES_fname );
}

void precon_schur_params_print( FILE *fid, Tprecon_schur_params_ptr p )
{
        fprintf( fid, "initial decomp %d\nnlevels %d\n", p->initial_decomp, p->nlevels );
        fprintf( fid, "lfill %d\ndroptol %g\n", p->lfill, p->droptol );
        fprintf( fid, "B preconditioner in file %s is ", p->precon_fname_B );
        precon_print_name( fid, p->precon_type_B );
        fprintf( fid, "S preconditioner in file %s is ", p->precon_fname_S );
        precon_print_name( fid, p->precon_type_S );
        fprintf( fid, "GMRES %d in file %s\n\n", p->GMRES, p->GMRES_fname );
}

void find_vtxdist( int n, TMPI_dat_ptr This, int block, int *vtxdist, int *vtxdist_b, int *vtxspace )
{
        int i; 

        MPI_Allgather( &n, 1, MPI_INT, vtxdist+1, 1, MPI_INT, This->comm );
        vtxdist[0] = 0;
        for( i=0; i<This->n_proc; i++ )
                vtxdist[i+1] = vtxdist[i+1] + vtxdist[i];
        memcpy( vtxdist_b, vtxdist, sizeof(int)*(This->n_proc+1) );
        if( block )
                for( i=1; i<This->n_proc+1; i++ )
                        vtxdist_b[i] *= BLOCK_SIZE;
        for( i=0; i<This->n_proc; i++ )
                vtxspace[i] = vtxdist_b[i+1] - vtxdist_b[i];
}

int unpacked_test_col( double *sdat, int n_bnd, int col ) 
{
        int ii, ee=0, eers[50];
        double *copy_from;

        copy_from = sdat;
        for( ii=0; ii<n_bnd; ii++, copy_from+=2 )
                if( !( finite(copy_from[0])&&finite(copy_from[1])&&finite(copy_from[n_bnd*2])&&finite(copy_from[n_bnd*2+1]) ) )
                        eers[ee++]=ii;
        if( ee )
        {
                printf( "\n%d errors in column %d\n", ee, col );
                for( ii=0; ii<ee; ii++ ){ 
                        copy_from = sdat + eers[ii]*2;
                        printf( "%d\n%15g\t%15g\n%15g\t%15g\n", eers[ii], copy_from[0], copy_from[1], copy_from[n_bnd*2], copy_from[n_bnd*2+1] );
                }
        }

        return ee;
}
