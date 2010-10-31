#include "parms_wrapper.h"
#include <parms.h>
#include <parms_map.h>
#include <parms_mat.h>
#include <parms_viewer.h>
#include <assert.h>

// Map
// ------

/*
void* parms_wrapperMapCreateFromDist(int *vtxdist, int *part, MPI_Comm mpicomm, int dof)
{
    parms_Map* map = malloc(sizeof(parms_Map));
    int flag = parms_MapCreateFromDist(map, vtxdist, part, mpicomm, 0, dof, INTERLACED);
    assert(flag == 0);
    return map;
}
*/

void* parms_wrapperMapCreateFromPtr(int gsize, int* nodes, int* vtxdist, MPI_Comm mpicomm, int dof)
{
    parms_Map* map = malloc(sizeof(parms_Map));
    assert(map);
    int flag = parms_MapCreateFromPtr(map, gsize, nodes, vtxdist, mpicomm, dof, INTERLACED);
    assert(flag == 0);
    return map;
}

void parms_wrapperMapFree(void* map)
{
    int flag = parms_MapFree(map);
    assert(flag == 0);
    free(map);
}

int parms_wrapperMapGetGlobalSize(void* map)
{
    return parms_MapGetGlobalSize(*(parms_Map*)map);
}
int parms_wrapperMapGetLocalSize(void* map)
{
    return parms_MapGetLocalSize(*(parms_Map*)map);
}

void parms_wrapperMapView(void* map, void* v)
{
    int flag = parms_MapView(*(parms_Map*)map, *(parms_Viewer*)v);
    assert(flag == 0);
}

// Matrix
// ------


void* parms_wrapperMatCreate(void* map)
{
    parms_Mat* mat = malloc(sizeof(parms_Mat));
    assert(mat);
    int flag = parms_MatCreate(mat, *(parms_Map*)map);
    assert(flag == 0);
    return mat;
}

void parms_wrapperMatFree(void* mat)
{
    int flag = parms_MatFree(mat);
    assert(flag == 0);
}

void parms_wrapperMatSetValues(void* mat, int m, int* im, int* ia, int* ja, double* values)
{
    int flag = parms_MatSetValues(*(parms_Mat*)mat, m, im, ia, ja, values, INSERT);
    assert(flag == 0);
}

void parms_wrapperMatSetup(void* mat)
{
    int flag = parms_MatSetup(*(parms_Mat*)mat);
    assert(flag == 0);
}

void parms_wrapperMatView(void* mat, void* viewer)
{
    int flag = parms_MatView(*(parms_Mat*)mat, *(parms_Viewer*)viewer);
    assert(flag == 0);
}

void parms_wrapperMatVec(void* mat, double *x, double *y)
{
    int flag = parms_MatVec(*(parms_Mat*)mat, x, y);
    assert(flag == 0);
}

// Preconditioner
// --------------

void* parms_wrapperPCCreate(void* mat)
{
    parms_PC* pc = malloc(sizeof(parms_PC));
    assert(pc);
    int flag = parms_PCCreate(pc, *(parms_Mat*)mat);
    assert(flag == 0);
    return pc;
}

char* parms_wrapperPCGetName(void* pc)
{
    char* p;
    int flag = parms_PCGetName(*(parms_PC*)pc, &p);
    assert(flag == 0);
    return p;
}

void parms_wrapperPCFree(void* pc)
{
    int flag = parms_PCFree(pc);
    assert(flag == 0);
}

void parms_wrapperPCSetType(void* pc, enum PCType pc_type)
{
    PCTYPE pctype;
    switch(pc_type) {
    case block_jacobi:
        pctype = PCBJ;
        break;
    case additive_schwarz:
        pctype = PCRAS;
        break;
    case schur_complement:
        pctype = PCSCHUR;
        break;
    }
    int flag = parms_PCSetType(*(parms_PC*)pc, pctype);
    assert(flag == 0);
}

void parms_wrapperPCSetup(void* pc)
{
    int flag = parms_PCSetup(*(parms_PC*)pc);
    assert(flag == 0);
}

void parms_wrapperPCView(void* pc, void* viewer)
{
    parms_PCView(*(parms_PC*)pc, *(parms_Viewer*)viewer);
//    assert(flag == 0);
}

void parms_wrapperPCApply(void* pc, double* y, double* x)
{
    int flag = parms_PCApply(*(parms_PC*)pc, y, x);
    assert(flag == 0);
}

void parms_wrapperSetILUType(void* pc, enum PCILUType pc_type)
{
    PCILUTYPE pctype;
    switch(pc_type) {
    case ilu0:
        pctype = PCILU0;
        break;
    case iluk:
        pctype = PCILUK;
        break;
    case ilut:
        pctype = PCILUT;
        break;
    case arms:
        pctype = PCARMS;
        break;
    }

    int flag = parms_PCSetILUType(*(parms_PC*)pc, pctype);
    assert(flag == 0);
}

void parms_wrapperPCSetTol(void* pc, double *tol)
{
    int flag = parms_PCSetTol(*(parms_PC*)pc, tol);
    assert(flag == 0);
}

void parms_wrapperPCSetNLevels(void* pc, int nlevel)
{
    int flag = parms_PCSetNlevels(*(parms_PC*)pc, nlevel);
    assert(flag == 0);
}

// Viewer
// ------

void* parms_wrapperViewerCreate(char* filename)
{
    parms_Viewer* v = malloc(sizeof(parms_Viewer));
    assert(v);
    int flag = parms_ViewerCreate(v, filename);
    assert(flag == 0);
    return v;
}

void parms_wrapperViewerFree(void* v)
{
    int flag = parms_ViewerFree(v);
    assert(flag == 0);
    free(v);
}
