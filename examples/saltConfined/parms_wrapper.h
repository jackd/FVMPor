#ifndef PARMS_WRAPPER_H
#define PARMS_WRAPPER_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

enum PCType {block_jacobi, additive_schwarz, schur_complement};
enum PCILUType {ilu0, iluk, ilut, arms};

// Map functions
//void* parms_wrapperMapCreateFromDist(int *vtxdist, int *part, MPI_Comm mpicomm, int dof);
void* parms_wrapperMapCreateFromPtr(int gsize, int* nodes, int* vtxdist, MPI_Comm mpicomm, int dof);
void parms_wrapperMapFree(void* map);
int parms_wrapperMapGetGlobalSize(void* map);
int parms_wrapperMapGetLocalSize(void* map);
void parms_wrapperMapView(void* map, void* viewer);

// Matrix functions
void* parms_wrapperMatCreate(void* map);
void parms_wrapperMatFree(void* mat);
void parms_wrapperMatSetValues(void* mat, int m, int* im, int* ia, int* ja, double* values);
void parms_wrapperMatSetup(void* mat);
void parms_wrapperMatView(void* mat, void* viewer);
void parms_wrapperMatVec(void* mat, double *x, double *y);

// Preconditioner functions
void* parms_wrapperPCCreate(void* mat);
char* parms_wrapperPCGetName(void* pc);
void parms_wrapperPCSetType(void* pc, enum PCType pc_type);
void parms_wrapperPCSetup(void* pc);
void parms_wrapperPCApply(void* pc, double* y, double* x);
void parms_wrapperPCView(void* pc, void* viewer);
void parms_wrapperPCFree(void* pc);
void parms_wrapperSetILUType(void* pc, enum PCILUType pctype);
void parms_wrapperPCSetTol(void* pc, double *tol);
void parms_wrapperPCSetNLevels(void* pc, int nlevel);

// Viewer functions
void* parms_wrapperViewerCreate(char* filename);
void parms_wrapperViewerFree(void* viewer);

#ifdef __cplusplus
}
#endif

#endif
