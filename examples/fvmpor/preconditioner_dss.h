#ifndef PRECONDITIONER_DSS_H
#define PRECONDITIONER_DSS_H

#include "fvmpor_ODE.h"

#include <fvm/preconditioner_base.h>

#include <vector>
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

    Preconditioner() : num_setups(0), num_callbacks(0), num_applications(0) {}

private:
    // universal to preconditioner implementations
    int num_setups;
    int num_callbacks;
    int num_applications;
    int N;

    int blocksize;
    ColumnPattern pat;

    std::vector<int> row_index;
    std::vector<int> columns;
    std::vector<double> values;

    std::vector<int> colourvec;
    int num_colours;

    std::vector<double> shift;

    void initialise(const mesh::Mesh& m);

    // unique to this implementation implementations
    _MKL_DSS_HANDLE_t dss_handle;
    int nnz;
};

} // end namespace fvmpor

#endif
