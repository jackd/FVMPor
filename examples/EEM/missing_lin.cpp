#include <lin/lin.h>
#include <algorithm>
#include <math.h>
#include <cassert>

using namespace lin;

typedef Vector<double> DVector;
typedef Matrix<double> DMatrix;

DMatrix subm(const DMatrix& A, int i1, int i2, int j1, int j2){
	assert(i1 > 0 && i2 <= A.rows() && j1 > 0 && j2 <= A.cols());
	DMatrix temp(i2-i1+1,j2-j1+1);
	for (int i = 1; i <= i2 - i1 + 1; ++i){
		for (int j = 1; j <= j2 - j1 + 1; ++j){
			temp(i,j) = A(i + i1 - 1, j + j1 - 1);
		}
	}
	return temp;
}

DVector subv(const DVector& a, int i1, int i2){
	assert(i1 > 0 && i2 <= a.size());
	DVector temp(i2 - i1);
	for (int i = 1; i <= i2 - i1 + 1; ++i){
		temp(i) = a(i + i1 - 1);
	}
	return temp;
}

DVector subv(const DMatrix& A, int i1, int i2, int j){
	assert(i1 > 0 && i2 <= A.rows() && j > 0 && j <= A.cols());
	DVector temp(i2 - i1 + 1);
	for (int i = 1; i <= i2 - i1 + 1; ++i){
		temp(i) = A(i + i1 - 1,j);
	}
	return temp;
}

DMatrix eye(int n){
    DMatrix A(n,n);
    for (int i = 1; i <= n; ++i){
        A(i,i) = 1;
    }
    return A;
}

void getPadeCoefficients(int m, double* c){
    switch (m){
        case 3:
            c[0] = 120.0;	c[1] = 60.0;	c[2] = 12.0;	c[3] = 1.0;
        case 5:
            c[0] = 30240.0; c[1] = 15120.0; c[2] = 3360.0; c[3] = 420.0; c[4] = 30.0; c[5] = 1.0;
        
        case 7:
            c[0] = 17297280.0; c[1] = 8648640.0; c[2] = 1995840.0; c[3] = 277200.0; c[4] = 25200.0; 
            c[5] = 1512.0;	   c[6] = 56.0;		 c[7] = 1.0;
            
        case 9:
            c[0] = 17643225600.0; c[1] = 8821612800.0; c[1] = 2075673600.0;  c[3] = 302702400.0;
            c[4] = 30270240.0;    c[5] = 2162160.0;    c[6] = 110880.0;      c[7] = 3960.0; 
            c[8] = 90.0;          c[9] = 1.0;
                            
        case 13:
            c[0] = 64764752532480000.0; c[1] = 32382376266240000.0; c[2] = 7771770303897600.0;
            c[3] = 1187353796428800.0;  c[4] = 129060195264000.0;   c[5] = 10559470521600.0;
            c[6] = 670442572800.0;      c[7] = 33522128640.0;       c[8] = 1323241920.0;
            c[9] = 40840800.0;          c[10] = 960960.0;           c[11] = 16380.0;
            c[12] = 182.0;              c[13] = 1.0;
    }
}

DMatrix operator+(DMatrix A, const DMatrix& B){
    int n = A.cols();
    int m = A.rows();
    assert((B.cols() == n) && (B.rows() == m));
    for (int i = 1; i <= n; ++i){
        for (int j = 1; j <= m; ++j){
            A(i,j) += B(i,j);
        }
    }
    return A;
}

DVector operator+(DVector a, const DVector& b){
	for (int i = 1; i <= a.size(); ++i){
		a(i) += b(i);
	}
	return a;
}

DVector plus(const DVector& a, const DVector&b){
	DVector temp(a.size());
	for (int i = 1; i <= a.size(); ++i){
		temp(i) = a(i) + b(i);
	}
	return temp;
}

DMatrix operator*(const DMatrix& A, const DMatrix& B){
//Matrix-matrix multiplication
    int n = A.cols();
    int m = A.rows();
    int nb = B.cols();
    int mb = B.rows();
    assert(m == nb);
    DMatrix prod(n,mb);
    for (int i = 1; i <= n; ++i){
        for (int j = 1; j <= m; ++j){
            for (int k = 1; k <= n; ++k){
                prod(i,j) += A(i,k)*B(k,j);
            }
        }
    }
    return prod;
}

DVector operator*(const DMatrix& A, const DVector& x)
//Matrix-vector multiplication
{
    assert(A.cols() == x.size());
    int n = A.rows();
    int mm = A.cols();
    DVector temp(n);
    for (int i = 1; i < n + 1; i++)
    {
        for (int j = 1; j < mm + 1; j++)
        {
            temp(i) += A(i,j)*x(j);
        }
    }
    return temp;
}

DMatrix operator*(double B, const DMatrix&  A)
//Scalar-matrix multiplication
{
    int n = A.rows();
    int mm = A.cols();
    DMatrix temp = A;
    for (int i = 1; i < n + 1; i++)
    {
        for (int j = 1; j < mm + 1; j++)
        {
            temp(i,j) *= B;
        }
    }
    return temp;
}

DVector operator*(double b, DVector a){
	for (int i = 1; i <= a.size(); ++i){
		a(i) *= b;
	}
	return a;
}

double dot(const DVector& x, const DVector& y){
    double temp = 0.0;
	int n = x.size();
    assert(n == y.size());
    for (int i = 1; i <= n; ++i){
        temp += x(i)*y(i);
        }
    return temp;
}

double norm(const DVector& x){
    return std::sqrt(dot(x,x));
}

DMatrix transpose(const DMatrix& A){
    int n = A.rows();
    int mm = A.cols();
    DMatrix temp(mm,n);
    for (int i = 1; i < n + 1; i++){
        for (int j = 1; j < mm + 1; j++){
            temp(j,i) = A(i,j);
        }
    }
    return temp;
}

double min(double a, double b){
    if (a > b){
        return b;
    }else{
        return a;
    }
}

double max(double a, double b){
    if (a < b){
        return b;
    }else{
        return a;
    }
}

int min(int a, int b){
    if (a > b){
        return b;
    }else{
        return a;
    }
}
	
/*
DMatrix phipade(const DMatrix& H) {
    assert(H.rows() == H.cols());

    int ideg = 6;   // degree of polynomial
    double t = 1.0; // compute phi(t*H) where t = 1

    // Form B = [H I ; 0 0]
    int mm = H.rows();
    DMatrix B(2*mm, 2*mm);
    B(1,mm, 1,mm) = H(1,mm,1,mm);
    
    for (int i = 1; i < mm + 1; i++){
        B(i,mm + i) = 1;
    }

    // Make call to dgpadm
    double* Bdata = const_cast<double*>(&B.at(0,0));
    int ldb = 2*mm;
    int sz = 4*ldb*ldb+ideg+1;
    DVector wsp(sz);
    double* wsp_data = const_cast<double*>(&wsp.at(0));
    int lwsp = sz;
    lin::Vector<int> ipiv(2*mm);
    int* ipiv_data = const_cast<int*>(&ipiv.at(0));
    int iexph;
    int ns;
    int iflag;
    dgpadm_(&ideg, &ldb, &t, Bdata, &ldb, wsp_data, &lwsp, ipiv_data, &iexph, &ns, &iflag);
    assert(iflag == 0);

    DMatrix expB(wsp_data+iexph-1, wsp_data+iexph+ldb*ldb-1, B.rows());
    return(expB(1,mm,mm+1,2*mm));
}
*/

double max(DVector u){
    double m = u(1);
    int n = u.size();
    for (int i = 1; i <= n;++i){
        if (u(i) > m){
            m = u(i);
        }
    }
    return m;
}

double sum(const DVector& u){
    double s = 0.0;
    int n = u.size();
    for (int i = 1; i <= n; ++i){
        s += u(i);
    }
    return s;
}

DVector sum(const DMatrix& A){
    int n = A.rows();
    int m = A.cols();
    DVector s(m);
    for (int i = 1; i <= m; ++i){
        s(i) = sum(A(all,i));
    }
    return s;
}

double abs(double A){
    if (A < 0){
        A = -A;
    }
    return A;
}

DVector abs(DVector A){
	int n = A.size();
	for (int i = 1; i <= n; ++i){
		A(i) = abs(A(i));
	}
	return A;
}

DMatrix abs(DMatrix A){
    int n = A.rows();
    int m = A.cols();
    for (int i = 1; i <= n; ++i){
        for (int j = 1; j <= m; ++j){
            A(i,j) = abs(A(i,j));
        }
    }
    return A;
}

void swap_rows(DMatrix& A, int i1, int i2){
    int m = A.cols();
    for (int j = 1; j < m; ++j){
        double temp = A(i1,j);
        A(i1,j) = A(i2,j);
        A(i2,j) = temp;
    }
}

void swap_entries(Vector<int>& u, int i, int j){
    int m = u.size();
    int temp = u(i);
    u(i) = u(j);
    u(j) = temp;
}

int partially_pivot(DMatrix& A, int k){
    int n = A.rows();
    bool flag = 0;
    int j = 0;
    for (int i = k; i <= n; ++i){
        if (abs(A(i,k)) > abs(A(k,k))){
            j = i;
            flag = 1;
        }
    }
    if (flag){
        swap_rows(A,k,j);
    }
    return j;
}

DMatrix lu_factor(DMatrix A, Vector<int>& p){
    int n = A.cols();
    int m = A.rows();
    for (int i = 1; i <= n; ++i){
        p(i) = i;
    }
    for (int k = 1; k <= n-1; ++k){
        int q = partially_pivot(A,k);
        if (q!= 0){
            swap_entries(p,k,q);
        }
        for (int x = k+1; x <= n; ++x){
            A(x,k) = A(x, k) / A(k, k);
        }
        for (int i = k+1; i <= n; ++i){
            //upper triangular section of lu
            for (int j = k+1; j <= n; ++j){
                A(i,j) -= A(i,k)*A(k,j);
            }
        }
    }
    return A;
}

DVector backward_sub(const DMatrix& lu, const DVector& b, bool ones_flag){
    int n = b.size();
    DVector x = b;
    for (int i = n; i >= 1; --i){
        for (int j = n; j > i; --j){
            x(i) -= lu(i,j)*x(j);
        }
        if (!ones_flag){
            x(i) /= lu(i,i);
        }
    }
    return x;
}

DVector forward_sub(const DMatrix& lu, const DVector& b, bool ones_flag){
    int n = b.size();
    DVector x = b;
    for (int i = 1; i <= n; ++i){
        for (int j = 1; j < i; ++j){
            x(i) -= lu(i,j)*x(j);
        }
        //divide
        if (!ones_flag){
            x(i) /= lu(i,i);
        }
    }
    return x;
}

DVector lu_solve(const DMatrix& lu, DVector b){
    b = forward_sub(lu,b,1);
    b = backward_sub(lu,b,0);
    return b;
}

void permute(DVector& b, const Vector<int>& p){
    int n = b.size();
    DVector bnew(n);
    for (int i = 1; i <= n; i++){
        bnew(i) = b(p(i));
    }
    b = bnew;
}

DVector lin_solve(const DMatrix& A, DVector b){
    assert(A.rows() == b.size());
    Vector<int> p(A.rows());
    DMatrix lu = lu_factor(A,p);
    permute(b,p);
    b = lu_solve(lu,b);
    return b;
}

DMatrix lin_solve(const DMatrix& A, const DMatrix& B){
    int n = A.rows();
    int m = B.cols();
    DMatrix sol(n,m);
    for (int j = 1; j <= m; ++j){
        sol(all,j) = lin_solve(A,B(all,j));
    }
    return sol;
}
            
DMatrix PadeApproximantOfDegree(int m, const DMatrix& A){
	int n = A.cols();
    assert(A.cols() == A.rows());
    double c[m];
    getPadeCoefficients(m,c);
    DMatrix U(n,n);
    DMatrix V(n,n);
	if (m != 13){
        DMatrix Apowers[m/2 + 1];
        Apowers[0] = eye(n);
		DMatrix A2 = A*A;
		/***FIXME**  for some reason I sometimes get a segmentation 
		fault when I try Apowers[1] = A*A;*/
        Apowers[1] = A2;
        for (int i = 2; i <= m/2; ++i){
            Apowers[i] = Apowers[i-1]*Apowers[1];
        }
        for (int j = m+1; j >= 2; ----j){
            U += c[j-1]*Apowers[j/2 - 1];
            
        }
        U = A*U;
        for (int j = m; j >= 1; ----j){
            V += c[j-1]*Apowers[(j+1)/2-1];
        }
    }
    else{ //case for m = 13
        DMatrix A2(n,m);
        DMatrix A4(n,m);
        DMatrix A6(n,m);
        
        A2 = A*A;
        A4 = A2*A2;
        A6 = A4*A2;  
        
        DMatrix I = eye(n);
        U = A * (A6*(c[13]*A6 + c[11]*A4 + c[9]*A2) + c[7]*A6 + c[5]*A4
                        + c[3]*A2 + c[1]*I);
                        
        V = A6*(c[12]*A6 + c[10]*A4 + c[8]*A2) + c[6]*A6 + c[4]*A4 
                    + c[2]*A2 + c[0]*I;
    }
    DMatrix F = lin_solve(-1*U + V, U + V);
    return F;
}  
            
DMatrix expm(const DMatrix& A){
    int m_vals[5] = {3, 5, 7, 9, 13};
    double theta[5] = { 0.01495585217958292, 
                        0.2539398330063230, 
                        0.9504178996162932, 
                        2.097847961257068, 
                        5.371920351148152};
    
    abs(A);
    sum(A);
    //max(A);
    double normA = max(sum(abs(A)));
    double nAot = normA/theta[4];
    DMatrix F(A.rows(),A.cols());
	
    if (nAot <= 1){
    //no scaling required
        for (int i = 0; i < 5; i++){
            if (normA <= theta[i]){
				F = PadeApproximantOfDegree(m_vals[i],A);
				}
        }
    }
    else{
        int s = 1;
        int s_exp = 2;
        while (s_exp < nAot){
            ++s;
            s_exp *= 2;
        }
        DMatrix Atemp = (1.0/s_exp)*A;
		F = PadeApproximantOfDegree(m_vals[4],Atemp);
		for (int i = 1; i <= s; ++i){
            F = F*F; //squaring
        }
    }
	return F;
} 

DMatrix phipade(const DMatrix& H){
	int n = H.rows();
	assert(H.rows() == H.cols());
	DMatrix B(2*n,2*n);
	//B(1,n,1,n) = H;//**FIXME**
	for (int i = 1; i <= n; ++i){
		for (int j = 1; j <= n; ++j){
			B(i,j) = H(i,j);
		}
	}
	for (int i = 1; i <= n; ++i){
		B(i,n+i) = 1;
	}
	
	DMatrix expB = expm(B);
	DMatrix phiB(n,n);
	//phiB = expB(1,n,n+1,2*n);
	for (int i = 1; i <= n; ++i){
		for (int j = 1; j <= n; ++j){
			phiB(i,j) = expB(i,j+n);
		}
	}
	return phiB;
}
