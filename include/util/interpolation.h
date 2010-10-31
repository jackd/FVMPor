#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#ifndef USE_MINLIN
#include "interpolation_legacy.h"
#else
#include <cuda_runtime.h>
#include <cusparse.h>

#include <lin/lin.h>
#include <lin/coordinators/gpu/coordinator.h>

#include <util/mklallocator.h>
#include <mkl_spblas.h>
#include <mkl_service.h>

#include <algorithm>

namespace util{

enum sparse_file_format {file_format_matlab};

template <typename T>
struct CoordTraits{
    static bool is_device() {return false;};
};
template <>
struct CoordTraits<lin::gpu::Coordinator<int> >{
    static bool is_device() {return true;};
};


// Interpolation matrix. In the most general sense this type is used
// for weighted linear maps from one vector to another.
// It is templated on a coordinator that specifies
// whether the matrix is stored on host or device memory.
//
// Initialised with index and weight vectors defined on the host memory
// If a GPU coordinator is specified, then a CUSP-based sparse matrix
// is allocated on device/GPU memory and used for interpolation
//template <typename TVec, typename TIndexVec, typename Coord>
template <typename Coord>
class InterpolationMatrix{
    typedef typename lin::rebind<Coord,int>::type CoordType;
    typedef typename lin::rebind<Coord,double>::type CoordWeights;
    typedef typename lin::rebind<Coord,int>::type CoordIndex;
    typedef typename lin::Vector<double,CoordWeights> TVec;
    typedef typename lin::Vector<int,CoordIndex> TIndexVec;
    typedef typename lin::Vector<double,lin::DefaultCoordinator<double> > TVecHost;
    typedef typename lin::Vector<int,lin::DefaultCoordinator<int> > TIndexVecHost;
public:
    InterpolationMatrix(){};
    InterpolationMatrix( TIndexVecHost& ia, TIndexVecHost& ja, TVecHost& v ):
        ia_(ia), ja_(ja), v_(v)
    {
        n_rows_ = ia_.size() - 1;
        nnz_ = ja_.size();
        n_cols_ = *(std::max_element(ja.begin(),ja.end())) + 1;
        // GPU : use CUSPARSE
        if( CoordTraits<CoordType>::is_device() ){
            assert( cusparseCreate(&handle_) == CUSPARSE_STATUS_SUCCESS );
            status_ = cusparseCreateMatDescr(&descra_);
            assert( status_==CUSPARSE_STATUS_SUCCESS );
            cusparseSetMatType(descra_, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(descra_, CUSPARSE_INDEX_BASE_ZERO);
        }
    }
    int rows() const{
        return n_rows_;
    }
    int cols() const{
        return n_cols_;
    }
    int nonzeros() const{
        return nnz_;
    }
    void matvec( const TVec& x, TVec& y ){
        double *x_ptr = const_cast<double*>(x.data());
        double *y_ptr = const_cast<double*>(y.data());
        int *row_ptr = const_cast<int*>(ia_.data());
        int *col_ptr = const_cast<int*>(ja_.data());
        double *v_ptr = const_cast<double*>(v_.data());
        assert(x.dim()>=n_cols_);
        assert(y.dim()==n_rows_);
        // GPU : use CUSPARSE
        if( CoordTraits<CoordType>::is_device() ){
            status_ = cusparseDcsrmv( handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, n_rows_, n_cols_, 1., descra_,
                                      v_ptr, row_ptr, col_ptr, x_ptr, 0., y_ptr);
            assert( status_==CUSPARSE_STATUS_SUCCESS );
        }
        // CPU : use MKL
        else
        {
            char transa = 'N';
            // uses zero-based cspblas version of the MKL function
            mkl_cspblas_dcsrgemv(&transa, &n_rows_, v_ptr, row_ptr, col_ptr, x_ptr, y_ptr );
        }
    }
    void matvec( const TVec& x, double* y_ptr ){
        double *x_ptr = const_cast<double*>(x.data());
        int *row_ptr = const_cast<int*>(ia_.data());
        int *col_ptr = const_cast<int*>(ja_.data());
        double *v_ptr = const_cast<double*>(v_.data());
        assert(x.dim()>=n_cols_);
        assert(y_ptr!=0);
        // GPU : use CUSPARSE
        if( CoordTraits<CoordType>::is_device() ){
            status_ = cusparseDcsrmv( handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, n_rows_, n_cols_, 1., descra_,
                                      v_ptr, row_ptr, col_ptr, x_ptr, 0., y_ptr);
            assert( status_==CUSPARSE_STATUS_SUCCESS );
        }
        // CPU : use MKL
        else
        {
            char transa = 'N';
            // uses zero-based cspblas version of the MKL function
            mkl_cspblas_dcsrgemv(&transa, &n_rows_, v_ptr, row_ptr, col_ptr, x_ptr, y_ptr );
        }
    }

    void write_to_file(std::string fname, sparse_file_format format){
        std::ofstream fid;
        fid.open(fname.c_str());
        switch(format){
            case file_format_matlab:
                fid << "ijv = [ ";
                for(int row=0; row<ia_.dim()-1; row++)
                    for(int j=ia_[row]; j<ia_[row+1]; j++)
                        fid << row+1 << " " << ja_[j]+1 << " " << v_[j] << ";";
                fid << "];" << std::endl;
                fid << "A = sparse(ijv(:,1), ijv(:,2), ijv(:,3));" << std::endl;
                break;
        }
        fid.close();
    }

    const TIndexVec& row_ptrs(){
        return ia_;
    }
    const TIndexVec& col_indexes(){
        return ja_;
    }

private:
    TIndexVec ia_;
    TIndexVec ja_;
    TVec v_;

    // for the sparse cuda calls
    cusparseStatus_t status_;
    cusparseHandle_t handle_;
    cusparseMatDescr_t descra_;

    // interpolation dimensions
    int n_cols_;
    int n_rows_;
    int nnz_;
};

}
#endif
#endif
