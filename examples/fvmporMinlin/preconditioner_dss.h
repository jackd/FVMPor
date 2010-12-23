#ifndef PRECONDITIONER_DSS_H
#define PRECONDITIONER_DSS_H

#include "fvmpor_ODE.h"

#include <fvm/preconditioner_base.h>

#include <vector>
#include <mkl_dss.h>

namespace fvmpor {

typedef std::vector< std::vector<int> > ColumnPattern;

class Preconditioner : public fvm::PreconditionerBase<Physics> {
    typedef fvm::PreconditionerBase<Physics> base;
    typedef base::TVecDevice TVecDevice;
    typedef base::TVec TVecHost;
    typedef TVecDevice::coordinator_type CoordDevice;
    typedef TVecHost::coordinator_type CoordHost;
    typedef lin::rebind<CoordDevice, int>::type CoordDeviceIndex;
    typedef lin::rebind<CoordHost, int>::type CoordHostIndex;
    typedef lin::Vector<int, CoordHostIndex> TVecHostIndex;
    typedef lin::Vector<int, CoordDeviceIndex> TVecDeviceIndex;

public:
    int setup(const mesh::Mesh& m, double tt, double c, double h,
              const TVecDevice &residual, const TVecDevice &weights,
              TVecDevice &sol, TVecDevice &derivative,
              TVecDevice &temp1, TVecDevice &temp2, TVecDevice &temp3,
              Callback compute_residual);

    int apply(const mesh::Mesh& m,
              double t, double c, double h, double delta,
              const TVecDevice &residual, const TVecDevice &weights,
              const TVecDevice &rhs,
              TVecDevice &sol, TVecDevice &derivative,
              TVecDevice &z, TVecDevice &temp,
              Callback compute_residual);

    int setups() const { return num_setups_; }
    int callbacks() const { return num_callbacks_; }
    int applications() const { return num_applications_; }
    double time_apply() {return time_apply_;};
    double time_compute() {return time_M_ + time_J_;};
    double time_jacobian() {return time_J_;};
    double time_M() {return time_M_;};

    Preconditioner() : num_setups_(0), num_callbacks_(0), num_applications_(0), time_M_(0), time_J_(0), time_apply_(0) {}

    void initialise(const mesh::Mesh& m);

private:
    // universal to preconditioner implementations
    double time_apply_;
    double time_J_;
    double time_M_;

    int num_setups_;
    int num_callbacks_;
    int num_applications_;
    int N_;

    int blocksize_;
    ColumnPattern pat_;

    std::vector<int> row_index_;
    std::vector<int> columns_;

    TVecHost values_;

    // here are the new variables that we are creating
    std::vector<TVecDeviceIndex> colour_p_; // list of the columns associated with each colour
    TVecHostIndex matrix_p_;
    std::vector<TVecDeviceIndex> res_p_; // the entries from the residual vector that are used
                                          // to compute nonzero entries in the jacobian
    std::vector<TVecDeviceIndex> shift_p_; // corresponding entries in shift vector
    TVecHostIndex colour_dist_;

    std::vector<int> colourvec_;
    int num_colours_;

    TVecDevice shift_;

    // unique to this implementation
    _MKL_DSS_HANDLE_t dss_handle_;
    int nnz_;
};

} // end namespace fvmpor

#endif
