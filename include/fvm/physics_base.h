#ifndef PHYSICS_BASE_H
#define PHYSICS_BASE_H

#include "fvm.h"
#include "mesh.h"

namespace fvm {

// A base class template for Physics classes.  It provides defaults for each of
// the required member functions.  Hence, derived classes need only replace
// those functions whose behaviour differs from the default.

template<typename Physics, typename ValueType>
class PhysicsBase {
public:
    typedef ValueType value_type;
    typedef typename fvm::Callback<Physics> Callback;
    typedef typename fvm::Iterator<value_type>::type iterator;
    typedef typename fvm::ConstIterator<value_type>::type const_iterator;

    void initialise(double& t,
                    const mesh::Mesh& m,
                    iterator sol,
                    iterator deriv,
                    iterator temp,
                    Callback compute_residual)
    {
        static_cast<Physics*>(this)->init(t, m, sol, deriv);
    }

    void init(double& t,
              const mesh::Mesh& m,
              iterator sol,
              iterator deriv)
    {
        // Initial condition is zero at time zero
    }

    void preprocess_timestep(double t,
                             const mesh::Mesh& m,
                             const_iterator sol,
                             const_iterator deriv)
    {
        // Do nothing
    }

    void postprocess_timestep(double t,
                              const mesh::Mesh& m,
                              const_iterator sol,
                              const_iterator deriv)
    {
        // Do nothing
    }

    void preprocess_evaluation(double t,
                               const mesh::Mesh& m,
                               const_iterator sol,
                               const_iterator deriv)
    {
        // Do nothing
    }

    void postprocess_evaluation(double t,
                                const mesh::Mesh& m,
                                const_iterator sol,
                                const_iterator deriv)
    {
        // Do nothing
    }

    value_type dirichlet(double t,
                         const mesh::Node& n)
    {
        // No Dirichlet nodes
        return value_type();
    }

    value_type lhs(double t,
                   const mesh::Volume& volume,
                   const_iterator sol,
                   const_iterator deriv)
    {
        // Left hand side is dy/dt
        return deriv[volume.id()];
    }

    value_type source(double t,
                      const mesh::Volume& volume,
                      const_iterator sol)
    {
        // No source
        return value_type();
    }

    value_type flux(double t,
                    const mesh::CVFace& cvf,
                    const_iterator sol)
    {
        // No flux
        return value_type();
    }

    value_type boundary_flux(double t,
                             const mesh::CVFace& cvf,
                             const_iterator sol)
    {
        // No flux
        return value_type();
    }

};

} // end namespace fvm

#endif
