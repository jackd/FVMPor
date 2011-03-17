#ifndef MISSING_LIN_H
    #define MISSING_LIN_H
	#include <lin/lin.h>

    typedef lin::Vector<double> DVector;
    typedef lin::Matrix<double> DMatrix;
	
	DMatrix subm(const DMatrix& A, int i1, int i2, int j1, int j2);
	DVector subv(const DVector& a, int i1, int i2);
	DVector subv(const DMatrix& A, int i1, int i2, int j);
	
	DVector operator*(double b, DVector a);
	DVector mul(double b, DVector a){return b*a;}
	
	DVector operator/(DVector a, double b){return a*(1/b);}
	
    DVector operator*(const DMatrix& A, const DVector& x);
	DVector mul(const DMatrix& A, const DVector& x){return A*x;}
	
    DMatrix operator*(double B, const DMatrix&  A);
	DMatrix mul(double B, const DMatrix& A){return B*A;}
	
	DVector operator+(DVector a, const DVector& b);
	DVector plus(const DVector& a, const DVector& b);
	
	DVector operator-(const DVector& a, const DVector& b){return a + (-1)*b;}
	DVector minus(const DVector& a, const DVector& b){return plus(a,mul(-1,b));}
	
	double dot(const DVector& x, const DVector& y);
    double norm(const DVector& x);
    DMatrix transpose(const DMatrix& A);
    double min(double a, double b);
	double max(double a, double b);
	//double max(DVector x);
    //int min(int a, int b);
    DMatrix phipade(const DMatrix& H);
    //DMatrix expm(const DMatrix& H);
    //DMatrix eye(int n);
    //DVector lin_solve(const DMatrix& A, DVector b);
	
#endif
