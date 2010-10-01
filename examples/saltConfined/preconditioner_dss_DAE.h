#ifndef PRECONDITIONER_DSS_DAE_H
#define PRECONDITIONER_DSS_DAE_H

#include "fvmpor_DAE.h"

#include <fvm/preconditioner_base.h>
#include <util/doublevector.h>

#include <mkl_dss.h>

namespace fvmpor {

typedef std::vector< std::vector<int> > ColumnPattern;

class Preconditioner : public fvm::PreconditionerBase<Physics> {
public:
    int setup(const mesh::Mesh& m, double tt, double c, double h,
              const_iterator residual, const_iterator weights,
              iterator sol, iterator derivative,
              iterator temp1, iterator temp2, iterator temp3,
              Callback compute_residual);

    int apply(const mesh::Mesh& m,
              double t, double c, double h, double delta,
              const_iterator residual, const_iterator weights,
              const_iterator rhs,
              iterator sol, iterator derivative,
              iterator z, iterator temp,
              Callback compute_residual);

    int setups() const { return num_setups; }
    int callbacks() const { return num_callbacks; }
    int applications() const { return num_applications; }
    const Physics& physics() const { return *my_physics; }

    Preconditioner(const Physics &physics_)  {
        num_setups = 0;
        num_callbacks = 0;
        num_applications = 0;
        my_physics = &physics_;
    }

private:
    const Physics *my_physics;
    DoubleVector D1, D2;

    int num_setups;
    int num_callbacks;
    int num_applications;

    int blocksize;
    int differential_blocksize, algebraic_blocksize;
    ColumnPattern pat;

    std::vector<int> row_index;
    std::vector<int> columns;
    std::vector<double> values;

    std::vector<int> colourvec;
    int num_colours;

    std::vector<double> shift;

    void initialise(const mesh::Mesh& m);

    // unique to this implementation
    _MKL_DSS_HANDLE_t dss_handle;
    int n, nnz;
};

} // end namespace fvmpor

#endif
