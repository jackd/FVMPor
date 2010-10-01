#include <scsl_blas.h>
#ifndef SGI
void daxpy(int n, double a, double* x, int incx, double* y, int incy) {
    daxpy_(&n, &a, x, &incx, y, &incy);
}
    
void dcopy(int n, double* x, int incx, double* y, int incy) {
    dcopy_(&n, x, &incx, y, &incy);
}

double ddot(int n, double* x, int incx, double* y, int incy) {
    return ddot_(&n, x, &incx, y, &incy);
}

double dnrm2(int n, double* x, int incx) {
    return dnrm2_(&n, x, &incx);
}

void drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    drot_(&n, x, &incx, y, &incy, &c, &s);
}

void drotg(double* a, double* b, double* c, double* s) {
    drotg_(a, b, c, s);
}

void dscal(int n, double a, double* x, int incx) {
    dscal_(&n, &a, x, &incx);
}

void dgemv(char* t, int m, int n, double alpha, double* a, int lda, double* x, int incx, double beta, double* y, int incy) {
    dgemv_(t, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void dger(int m, int n, double alpha, double* x, int incx, double* y, int incy, double* a, int lda) {
    dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

void dgemm(char* transa, char* transb, int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc) {
    dgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void dtrsm(char* side, char* uplo, char* transa, char* diag, int m, int n, double alpha, double* a, int lda, double* b, int ldb) {
    dtrsm_(side, uplo, transa, diag, &m, &n, &alpha, a, &lda, b, &ldb);
}
#endif
