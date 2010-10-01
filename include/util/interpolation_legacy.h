#ifndef INTERPOLATION_LEGACY_H
#define INTERPOLATION_LEGACY_H

#include <util/mklallocator.h>
#include <mkl_spblas.h>
#include <mkl_service.h>

namespace util{

template <typename TVec, typename TIndexVec>
class InterpolationMatrix{
public:
    InterpolationMatrix(){};
    InterpolationMatrix( TIndexVec& ia, TIndexVec& ja, TVec& v ):
        ia_(ia), ja_(ja), v_(v){
    }
    void matvec( const TVec& x, TVec& y ){
        int n = ia_.size() - 1;
#ifdef USE_MKL
        char transa = 'N';
        mkl_cspblas_dcsrgemv(&transa, &n,
                             const_cast<double*>(v_.data()),
                             const_cast<int*>(ia_.data()),
                             const_cast<int*>(ja_.data()),
                             const_cast<double*>(x.data()),
                             y.data());
#else
        for( int i=0; i<n; i++ ){
            y[i] = 0.;
            for( int j=ia_[i]; j<ia_[i+1]; j++ ){
                int col = ja_[j];
                y[i]+=v_[j]*x[col];
            }
        }
#endif
    }

    void write_to_file(std::string fname, sparse_file_format format){
        std::ofstream fid;
        fid.open(fname.c_str());
        switch(format){
            case file_format_matlab:
                fid << "ijv = [ ";
                for(int row=0; row<ia_.size()-1; row++)
                    for(int j=ia_[row]; j<ia_[row+1]; j++)
                        fid << row+1 << " " << ja_[j]+1 << " " << v_[j] << ";";
                fid << "];" << std::endl;
                fid << "A = sparse(ijv(:,1), ijv(:,2), ijv(:,3));" << std::endl;
                break;
        }
        fid.close();
    }

private:
    TIndexVec ia_;
    TIndexVec ja_;
    TVec v_;
};

}
#endif
