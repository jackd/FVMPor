/*
	benlib.h

	random stuff that everyone wants for christmas
*/

#ifndef __BENLIB_H__
#define __BENLIB_H__

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// comment out if you want to turn debugging code off
// only used in a haphazard manner any way, someone might clean up?
#define DEBUG

#define HEAD 2177
#define TAIL 3141

#define TAG_NODES 0
#define TAG_NNZ 1
#define TAG_INDEX 2
#define TAG_DAT 3
#define TAG_TAG 4
#define TAG_PART 5
#define TAG_NZ 6

// assert macro
#define  ASSERT(expr) 	\
			if( !(expr) ) 	\
			{					\
				fprintf(stderr,"ASSERT : failed on line %d of file %s : " #expr "\n", __LINE__, __FILE__ );	\
				MPI_Finalize(); 	\
				exit(1); \
			}
#define  ASSERT_MSG(expr,msg) 	\
			if( !(expr) ) 	\
			{					\
				fprintf(stderr,"ASSERT : failed on line %d of file %s : " #expr "\n", __LINE__, __FILE__ );	\
				fprintf(stderr,"%s\n",msg); \
				MPI_Finalize(); 	\
				exit(1); \
			}
#define  ASSERT_MPI(expr,msg,comm) \
			{ \
			int success, success_global; \
			success=expr;\
			if( !success ) 	\
				fprintf(stderr,"ASSERT : failed on line %d of file %s : " #expr " : %s\n", __LINE__, __FILE__, msg );	\
			MPI_Allreduce( &success, &success_global, 1, MPI_INT, MPI_LAND, comm ); \
			if( !success_global ){ MPI_Finalize();	exit(1);  }\
			}\

#define ERROR( msg ) 	fprintf( stderr, "ERROR : %s:%d\n%s\n", __FILE__, __LINE__, msg);		
#define WARNING( msg ) 	fprintf( stderr, "WARNING : %s:%d\n%s\n", __FILE__, __LINE__, msg);	

#define sWARNING( msg ) { char _errmsg[256]; msg; fprintf(stderr, "WARNING : %s:%d : %s\n", __FILE__, __LINE__,_errmsg); }
#define sERROR( msg )   { char _errmsg[256]; msg; fprintf(stderr, "ERROR : %s:%d : %s\n", __FILE__, __LINE__,  _errmsg); }
		

// warning command
//#define WARNING( str ) { fprintf(stderr,"\t\tWARNING : %s\n",str); }

// flags whether blocks are being used
#define BLOCKS 1

// the size of blocks in a block matrix
#define BLOCK_SIZE 2

// the macros below are custom operations for manipulating blocks of size
// 2. if you change BLOCK_SIZE then these must be changed as well.

#define LWORK BLOCK_SIZE*16

// the block matrix shift multiplier
#define BLOCK_M_SHIFT << 2

// the block vector shift multiplier
#define BLOCK_V_SHIFT << 1

// invert a block
#define BLOCK_M_INVERT( A ) { dgetrf_( &_BLOCK_SIZE, &_BLOCK_SIZE, A, &_BLOCK_SIZE, _IPIV, &_INFO ); dgetri_( &_BLOCK_SIZE, A, &_BLOCK_SIZE, _IPIV, _WORK, &_LWORK, &_INFO ); } 

// test if a vector block has nz entries
#define BLOCK_V_ISNZ( v ) ( (v)[0] || (v)[1] )

// inner product of two vector blocks
#define BLOCK_V_IP( u, v ) ( (u)[0]*(v)[0] + (v)[0]*(v)[1] )

// set all entries to the same scalar value
#define BLOCK_M_SET( a, A ) {(A)[0] = (A)[1] = (A)[2] = (A)[3] = (a);}

// set all entries in vector block to the same value
#define BLOCK_V_SET( a, v ) {(v)[0] = (v)[1]=(a); }

// block matrix copy
#define BLOCK_M_COPY( from, to ) { (to)[0]=(double)(from)[0]; (to)[1]=(double)(from)[1]; (to)[2]=(double)(from)[2]; (to)[3]=(double)(from)[3]; }

// block vector copy
#define BLOCK_V_COPY( from, to ) { (to)[0]=(double)(from)[0]; (to)[1]=(double)(from)[1]; }
#define BLOCK_V_ACOPY( a, from, to ) { (to)[0]=(a)*(double)(from)[0]; (to)[1]=(a)*(double)(from)[1]; }

// block matrix transpose
#define BLOCK_M_TRANSPOSE( A, tmp ) { (tmp) = (A)[1]; (A)[1]=(A)[2]; (A)[2]=(tmp); }

// block matrix addition a = b + c
#define BLOCK_M_ADD( a, b, c ) { (a)[0]=(b)[0]+(c)[0]; (a)[1]=(b)[1]+(c)[1]; (a)[2]=(b)[2]+(c)[2]; (a)[3]=(b)[3]+(c)[3]; }

// block matrix subtraction a = b - c
#define BLOCK_M_SUB( a, b, c ) { (a)[0]=(b)[0]-(c)[0]; (a)[1]=(b)[1]-(c)[1]; (a)[2]=(b)[2]-(c)[2]; (a)[3]=(b)[3]-(c)[3]; }

// block vector scale by scalar
#define BLOCK_V_SCALE(  s, v ) { (v)[0]*=(s); (v)[1]*=(s); }

// block vector addition a = b + c
#define BLOCK_V_ADD( a, b, c ) { (a)[0]=(b)[0]+(c)[0] ; (a)[1]=(b)[1]+(c)[1]; }

// block vector subtraction a = b - c
#define BLOCK_V_SUB( a, b, c ) { (a)[0]=(b)[0]-(c)[0] ; (a)[1]=(b)[1]-(c)[1]; }

// block vector axpy : u = a*x +y
#define BLOCK_V_AXPY( u, a, x, y ) { (u)[0] = (a)*(x)[0] + (y)[0]; (u)[1] = (a)*(x)[1] + (y)[1]; }

// block vector : u = a*x + b*u
#define BLOCK_V_AXBU( u, x, a, b ) { (u)[0] = (a)*(x)[0] + (b)*(u)[0]; (u)[1] = (a)*(x)[1] + (b)*(u)[1]; }

// block scalar multiplication X = a*X
#define BLOCK_M_SMUL( a, X ) { (X)[0]*=(a); (X)[1]*=(a); (X)[2]*=(a); (X)[3]*=(a); }

// block matrix multiply a = b * c
#define BLOCK_M_MULT( a, b, c ) { (a)[0]=(b)[0]*(c)[0]+(b)[2]*(c)[1]; (a)[2]=(b)[0]*(c)[2]+(b)[2]*(c)[3]; (a)[1]=(b)[1]*(c)[0]+(b)[3]*(c)[1]; (a)[3]=(b)[1]*(c)[2]+(b)[3]*(c)[3]; }

// LHS subtractive block matrix multiply a -= b * c
#define BLOCK_M_SUB_MULT( a, b, c ) { (a)[0]-=(b)[0]*(c)[0]+(b)[2]*(c)[1]; (a)[2]-=(b)[0]*(c)[2]+(b)[2]*(c)[3]; (a)[1]-=(b)[1]*(c)[0]+(b)[3]*(c)[1]; (a)[3]-=(b)[1]*(c)[2]+(b)[3]*(c)[3]; }

// block vector multiply a = b * c
#define BLOCK_MV_MULT( a, b, c ) { (a)[0]=(b)[0]*(c)[0]+(b)[2]*(c)[1]; (a)[1]=(b)[1]*(c)[0]+(b)[3]*(c)[1];}

// LHS subtractive block vector multiply a -= b * c
#define BLOCK_MV_SUB_MULT( a, b, c ) { (a)[0]-=(b)[0]*(c)[0]+(b)[2]*(c)[1]; (a)[1]-=(b)[1]*(c)[0]+(b)[3]*(c)[1];}

// block vector multiply, with a scalar s : a += s* B * c
#define BLOCK_MV_AMULT( a, b, c, s ) { (a)[0]+=(s)*((b)[0]*(c)[0]+(b)[2]*(c)[1]); (a)[1]+=(s)*((b)[1]*(c)[0]+(b)[3]*(c)[1]);}

#define BLOCK_M_PRINT( A ) {printf( "%g \t%g\n%g \t%g\n", (A)[0], (A)[2], (A)[1], (A)[3] );}

// find the sum of the columns in a block matrix, same as b=sum(A) in Matlab
#define BLOCK_M_COLSUM( A, b ) {(b)[0] = (A)[0]+(A)[1]; (b)[1] = (A)[2]+(A)[3];}

// find the sum of the squares of the columns in a block matrix, same as b=sum(A.^2) in Matlab
#define BLOCK_M_COLSUM_SQUARES( A, b ) {(b)[0] = (A)[0]*(A)[0]+(A)[1]*(A)[1]; (b)[1] = (A)[2]*(A)[2]+(A)[3]*(A)[3];}

// same as the above two macros, but is cumulative, as in b = b + sum(A); b = b + sum(A.^2)
#define BLOCK_M_COLSUM_CUM( A, b ) {(b)[0] += (A)[0]+(A)[1]; (b)[1] += (A)[2]+(A)[3];}
#define BLOCK_M_COLSUM_SQUARES_CUM( A, b ) {(b)[0] += (A)[0]*(A)[0]+(A)[1]*(A)[1]; (b)[1] += (A)[2]*(A)[2]+(A)[3]*(A)[3];}

// scale the columns of A by entires in b : A(:,j) op= b(j)
#define BLOCK_M_COLSCALE_MUL( A, b ) {(A)[0]*=(b)[0];(A)[1]*=(b)[0];(A)[2]*=(b)[1];(A)[3]*=(b)[1];}
#define BLOCK_M_COLSCALE_DIV( A, b ) {(A)[0]/=((b)[0]) ? (b)[0] : 1;(A)[1]/=((b)[0]) ? (b)[0] : 1;(A)[2]/=((b)[1]) ? (b)[1] : 1;(A)[3]/=((b)[1]) ? (b)[1] : 1;}



extern int _IPIV[BLOCK_SIZE];
extern double _WORK[LWORK];
extern int _INFO;
extern int _BLOCK_SIZE;
extern int _LWORK;
/*
	some data types
*/
typedef struct dnode
{
	struct dnode *next;
	struct dnode *prev;
	int n;
	int tag;
	int block;
	int *indx;
	double *dat;
} Tnode, *Tnode_ptr;

typedef struct node_list
{
	int n;
	int block;
	Tnode_ptr start;
	Tnode_ptr head;
	Tnode_ptr opt;
} Tnode_list, *Tnode_list_ptr;

/*
	function prototypes
*/

// sorting algorithms
extern void mergeSort(int numbers[], int temp[], int array_size);
extern void m_sort(int numbers[], int temp[], int left, int right);
extern void merge(int numbers[], int temp[], int left, int mid, int right);
extern void heapsort_int_index( int n, int *index, int *ra );
extern void heapsort_int( int n, int *ra);
extern void heapsort_double_index( int n, int *index, double *ra );
extern void heapsort_int_dindex( int n, double *index, int *ra );
extern void heapsort_longlong( int n, long long *ra);
extern void heapsort_longlong_index( int n, int *index, long long *ra );
extern void heapsort_double( int n, double *ra);

// searching algorithms
int binary_search(int n, int *array, int key);
int binary_search_double_bracket(int n, double *array, double key);

// permutation of arrays (scalar and block operations)
extern void permute( double *v, double *u, int *p, int n, int map_flag );
extern void permute_inplace( double *v, int *p, int n, int map_flag );
extern void permuteB( double *v, double *u, int *p, int n, int map_flag );

// takes an array of blocks and transposes the blocks
extern void block_transpose( double *a, int n );

// node and node_list algorithms
extern Tnode_ptr node_new( int n, int tag, int block, Tnode_ptr prev, Tnode_ptr next );
extern void node_list_free( Tnode_list_ptr L );
extern void node_list_push_head( Tnode_list_ptr L, Tnode_ptr node );
extern void node_list_push_start( Tnode_list_ptr L, Tnode_ptr node );
extern Tnode_ptr node_list_pop( Tnode_list_ptr L );
extern int  node_list_add( Tnode_list_ptr L, int n, int tag );
extern void node_list_init( Tnode_list_ptr L, int block );
extern int  node_list_verify( FILE *stream, Tnode_list_ptr L );
extern void print_nodes( FILE *stream, char *msg, Tnode_list_ptr L );

// timing code
extern double get_time( void );

#endif
