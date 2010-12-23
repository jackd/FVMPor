#ifndef FVM_H
#define FVM_H

// Template Parameter Requirements:
// --------------------------------

// The Physics template parameter represents the physics of the underlying
// PDE.  The PDE is assumed to take the form:

// f(u, du/dt) = -div(j) + s

// The requirements are as shown in this skeleton implementation.

/*
// This is the value stored at each node of the mesh.
// It can be a POD struct of double (one per dependent variable),
// or else it can be simply double (if there is only one variable)
struct MyValue {
    // e.g.
    static const int variables = 2;
    double pressure;
    double temperature;
};

class Physics {
public:

    // Whatever your value type, it must be typedefed to value_type
    typedef MyValue value_type;

    // Convenience typedefs
    typedef fvm::Iterator<value_type>::type iterator;
    typedef fvm::ConstIterator<value_type>::type const_iterator;

    void init(double& t,
              const mesh::Mesh& m,
              iterator sol,
              iterator deriv)
    {
        // This function is responsible for setting the solution to its
        // initial value.  It is also where any internal initialisation of
        // the physics should take place.

        // Parameters:
        // t:     the initial time
        // m:     the mesh
        // sol:   a random access iterator designating the beginning of the
        //        local solution vector
        // deriv: a random access iterator designating the beginning of the
        //        local derivative vector

        // Pre:
        // t is zero
        // [sol, sol + m.local_nodes()) is zero
        // [deriv, deriv + m.local_nodes()) is zero

        // Post:
        // t has been set to the initial time
        // [sol, sol + m.local_nodes()) has been set to the initial value
        // of the local solution
        // [deriv, deriv + m.local_nodes()) has been set to the initial
        // derivative of the local solution

        // Called:
        // Once.
        // This is the first member function of Physics to be called.
    }

    void initialise(double t,
                    const mesh::Mesh& m,
                    iterator sol,
                    iterator deriv,
                    iterator temp,
                    callback compute_residual)
    {
        // This function is responsible for setting the solution to its
        // initial value.  It is also where any internal initialisation of
        // the physics should take place.

        // Parameters:
        // t:     the initial time
        // m:     the mesh
        // sol:   a random access iterator designating the beginning of the
        //        local solution vector
        // deriv: a random access iterator designating the beginning of the
        //        local derivative vector
        // temp:  a random access iterator designating the beginning of
        //        temporary storage
        // compute_residual: residual function callback

        // Pre:
        // [sol, sol + m.local_nodes()) is zero
        // [deriv, deriv + m.local_nodes()) is zero
        // [temp, temp + m.local_nodes()) is zero

        // Post:
        // [sol, sol + m.local_nodes()) has been set to the initial value
        // of the local solution
        // [deriv, deriv + m.local_nodes()) has been set to the initial
        // derivative of the local solution

        // Called:
        // Once.
        // This is the first member function of Physics to be called.
    }

    void preprocess_timestep(double t,
                             const mesh::Mesh& m,
                             const_iterator sol,
                             const_iterator deriv)
    {
        // This function is responsible for any preprocessing that is
        // necessary before computation takes place for a new timestep.

        // Parameters:
        // t:     the current time
        // m:     the mesh
        // sol:   a random access iterator designating the beginning of the
        //        local solution vector
        // deriv: a random access iterator designating the beginning of the
        //        derivative of the local solution vector

        // Pre:
        // [sol, sol + m.local_nodes()) represents the current solution
        // [deriv, deriv + m.local_nodes()) represents its derivative

        // Called:
        // As the first call in advance
    }

    void postprocess_timestep(double t,
                              const mesh::Mesh& m,
                              const_iterator sol,
                              const_iterator deriv)
    {
        // This function is responsible for any postprocessing that is
        // necessary after computation takes place for a new timestep.

        // Parameters:
        // t:     the current time
        // m:     the mesh
        // sol:   a random access iterator designating the beginning of the
        //        local solution vector
        // deriv: a random access iterator designating the beginning of the
        //        derivative of the local solution vector

        // Pre:
        // [sol, sol + m.local_nodes()) represents the current solution
        // [deriv, deriv + m.local_nodes()) represents its derivative

        // Called:
        // As the last call in advance
    }

    void preprocess_evaluation(double t,
                               const mesh::Mesh& m,
                               const_iterator sol,
                               const_iterator deriv)
     {
        // This function is responsible for any preprocessing that is
        // necessary before evaluations of dirichlet, lhs, source, flux
        // and boundary_flux for the current solution iterate.

        // Parameters:
        // t:   the current time
        // m:   the mesh
        // sol:   a random access iterator designating the beginning of the
        //        current iterate of the solution vector
        // deriv: a random access iterator designating the beginning of the
        //        current iterate of the derivative of the solution vector

        // Pre:
        // [sol, sol + m.nodes()) represents the current solution iterate
        // [deriv, deriv + m.nodes()) represents its derivative

        // Called:
        // Before dirichlet, lhs, source, flux and boundary_flux for the
        // current solution iterate
    }

    void postprocess_evaluation(double t,
                                const mesh::Mesh& m,
                                const_iterator sol,
                                const_iterator deriv)
     {
        // This function is responsible for any postprocessing that is
        // necessary after evaluations of dirichlet, lhs, source, flux
        // and boundary_flux for the current solution iterate.

        // Parameters:
        // t:   the current time
        // m:   the mesh
        // sol:   a random access iterator designating the beginning of the
        //        current iterate of the solution vector
        // deriv: a random access iterator designating the beginning of the
        //        current iterate of the derivative of the solution vector

        // Pre:
        // [sol, sol + m.nodes()) represents the current solution iterate
        // [deriv, deriv + m.nodes()) represents its derivative

        // Called:
        // After dirichlet, lhs, source, flux and boundary_flux for the
        // current solution iterate
    }

    value_type dirichlet(double t,
                         const mesh::Node& n)
    {
        // This function is responsible for identifying any variables
        // which satisfy Dirichlet boundary conditions

        // Parameters:
        // t:            the current time
        // n:            the node

        // Post:
        // Let d denote the return value.
        // For each variable at the node in question, if that variable
        // satisfies a Dirichlet condition, then its field in d is set to
        // true (nonzero); otherwise it is set to false (zero).

        // Called:
        // Once per solution iterate for each boundary node.

    }

    value_type lhs(double t,
                   const mesh::Volume& volume,
                   const_iterator sol,
                   const_iterator deriv)
    {
        // This function is responsible for computing the control volume
        // averaged value of the left hand side f.

        // Parameters:
        // t:      the current time
        // volume: the control volume
        // sol:    a random access iterator designating the beginning of the
        //         current iterate of the solution vector
        // deriv: a random access iterator designating the beginning of the
        //        current iterate of the derivative of the solution vector

        // Pre:
        // [sol, sol + m.nodes()) represents the current solution iterate
        // [deriv, deriv + m.nodes()) represents its derivative

        // Post:
        // Return value is the control volume averaged value of the left
        // hand side f of the PDE

        // Called:
        // Once per solution iterate for each control volume in the mesh

    }

    value_type source(double t,
                      const mesh::Volume& volume,
                      const_iterator sol)
    {
        // This function is responsible for computing the control volume
        // averaged value of the source.

        // Parameters:
        // t:      the current time
        // volume: the control volume
        // sol:    a random access iterator designating the beginning of the
        //         current iterate of the solution vector

        // Pre:
        // [sol, sol + m.nodes()) represents the current solution iterate

        // Post:
        // Return value is the control volume averaged value of the source

        // Called:
        // Once per solution iterate for each control volume in the mesh
    }

    value_type flux(double t,
                    const mesh::CVFace& cvf,
                    const_iterator sol)
    {
        // This function is responsible for computing the flux through a
        // non-boundary control volume face.

        // Parameters:
        // t:   the current time
        // cvf: the control volume face
        // sol:    a random access iterator designating the beginning of the
        //         current iterate of the solution vector

        // Pre:
        // cvf.boundary() == 0
        // [sol, sol + m.nodes()) represents the current solution iterate

        // Post:
        // Return value is the flux through the control volume face in the
        // direction of cvf.unit_normal()

        // Called:
        // Once per solution iterate for each non-boundary control volume
        // face in the mesh
    }

    value_type boundary_flux(double t,
                             const mesh::CVFace& cvf,
                             const_iterator sol)
    {
        // This function is responsible for computing the flux through a
        // boundary control volume face.

        // Parameters:
        // t:   the current time
        // cvf: the control volume face
        // sol:    a random access iterator designating the beginning of the
        //         current iterate of the solution vector

        // Pre:
        // cvf.boundary() != 0
        // [sol, sol + m.nodes()) represents the current solution iterate

        // Post:
        // Return value is the flux through the control volume face in the
        // direction of cvf.unit_normal()

        // Called:
        // Once per solution iterate for each boundary control volume face
        // in the mesh
    }

};

*/

#include <util/checked_iterator.h>

namespace fvm {

using util::checked_iterator;

// Forward declaration
template<class Physics>
class SolverBase;

// Residual function callback
template<class Physics>
class Callback {
public:
    typedef typename Physics::TVecDevice TVecDevice;
    Callback() : solver() {};
    Callback(SolverBase<Physics>* solver) : solver(solver) {};
    // DEVICE
    int operator()(TVecDevice &y, bool communicate);
    //template<typename Iterator>
    //int operator()(Iterator it, bool communicate);
private:
    SolverBase<Physics>* solver;
};

// Simple traits class
template<typename T>
struct VariableTraits {
    enum {number = T::variables};
    enum {number_diff = T::differential_variables};
};
// And the specialisation for double
template<>
struct VariableTraits<double> {
    enum {number = 1};
    enum {number_diff = 1};
};

// The indirection through this struct is to force the caller of
// checked_iterator to be explicit about constness.
template<typename T>
struct pointer {
    typedef T* type;
};

// Iterator types.  In debug mode these are checked iterators.
template<typename T>
struct Iterator {
    #ifdef FVM_DEBUG
        typedef checked_iterator<T> type;
    #else
        typedef T* type;
    #endif
};

template<typename T>
struct ConstIterator {
    #ifdef FVM_DEBUG
        typedef checked_iterator<const T> type;
    #else
        typedef const T* type;
    #endif
};

// Convenience functions
#ifdef FVM_DEBUG
    template<typename T>
    checked_iterator<T> make_iterator(typename pointer<T>::type it,
                                      typename pointer<T>::type begin,
                                      typename pointer<T>::type end)
    {
        return checked_iterator<T>(it, begin, end);
    }
#else
    template<typename T>
    typename pointer<T>::type make_iterator(typename pointer<T>::type it,
                                            typename pointer<T>::type begin,
                                            typename pointer<T>::type end)
    {
        return it;
    }
#endif

} // end namespace fvm

#endif
