// the interface to the BLAS routines varies depeneding on the implementation on the particular architecture
// I am probably reinventing the wheele here
#ifdef SGI
#include <mkl.h>
#else
void daxpy(int n, double a, double* x, int incx, double* y, int incy);
void dcopy(int n, double* x, int incx, double* y, int incy);
double ddot(int n, double* x, int incx, double* y, int incy);
double dnrm2(int n, double* x, int incx);
void drot(int n, double* x, int incx, double* y, int incy, double c, double s);
void drotg(double* a, double* b, double* c, double* s);
void dscal(int n, double a, double* x, int incx);
void dgemv(char* t, int m, int n, double alpha, double* a, int lda, double* x, int incx, double beta, double* y, int incy);
void dger(int m, int n, double alpha, double* x, int incx, double* y, int incy, double* a, int lda);
void dgemm(char* transa, char* transb, int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc);
void dtrsm(char* side, char* uplo, char* transa, char* diag, int m, int n, double alpha, double* a, int lda, double* b, int ldb);

void daxpy_(int* n, double* a, double* x, int* incx, double* y, int* incy);
void dcopy_(int* n, double* x, int* incx, double* y, int* incy);
double ddot_(int* n, double* x, int* incx, double* y, int* incy);
double dnrm2_(int* n, double* x, int* incx);
void drot_(int* n, double* x, int* incx, double* y, int* incy, double* c, double* s);
void drotg_(double* a, double* b, double* c, double* s);
void dscal_(int* n, double* a, double* x, int* incx);
void dgemv_(char* t, int* m, int* n, double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy);
void dger_(int* m, int* n, double* alpha, double* x, int* incx, double* y, int* incy, double* a, int* lda);
void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);
void dtrsm_(char* side, char* uplo, char* transa, char* diag, int* m, int* n, double* alpha, double* a, int* lda, double* b, int* ldb);
#endif
