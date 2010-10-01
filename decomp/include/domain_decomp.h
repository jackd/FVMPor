#ifndef __DOMAIN_DECOMP_H__
#define __DOMAIN_DECOMP_H__

#include "linalg.h"
#include "indices.h"
#include "fileio.h"
#include "ben_mpi.h"

#define ELEMENT_TRIANGLE 2
#define ELEMENT_QUAD 3
#define ELEMENT_TETRAHEDRON 4
#define ELEMENT_HEXAHEDRON 5
#define ELEMENT_PRISM 6

typedef struct element
{
    int n_nodes;
    int type;
    int n_faces;
    int tag;

    int physical_tag;

    int *nodes;
    int *face_bcs;

    //int n_edges;
} Telement, *Telement_ptr;

typedef struct mesh
{
    int n_nodes;
    int n_elements;

    Telement_ptr elements;
    double *node_x;
    double *node_y;
    double *node_z;
    int    **node_bcs;
    int    *node_num_bcs;
    int **node_elements;
    int *node_num_elements;

    Tmtx_CRS A;
} Tmesh, *Tmesh_ptr;

void mesh_split( Tmesh_ptr mesh, TMPI_dat *This, int root );

#endif
