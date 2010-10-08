#ifndef FVM_ASSEMBLER_H
#define FVM_ASSEMBLER_H

#include <fvm/fvm.h>
#include <fvm/mesh.h>
#include <mpi/mpicomm.h>

#include <vector>

namespace fvm {

template<class Physics>
class FVMAssembler {
public:
    typedef mesh::Mesh Mesh;

    typedef typename Physics::value_type value_type;
    typedef typename Iterator<value_type>::type iterator;
    typedef typename ConstIterator<value_type>::type const_iterator;

    FVMAssembler(const Mesh& m, Physics& p);
    const Mesh& mesh() const;
    Physics& physics();
    template<typename Iterator>
    int compute_residual(
        double time, const_iterator u, const_iterator up, Iterator res);
private:
    FVMAssembler(const FVMAssembler&);
    FVMAssembler& operator=(const FVMAssembler&);

    const Mesh& m;
    Physics& p;
    mpi::MPICommPtr mpicomm_;

    std::vector<value_type> is_dirichlet;

    static const int variables_per_node = VariableTraits<value_type>::number;

    static void add_in_place(value_type&, value_type);
    static void subtract_in_place(value_type&, value_type);
    static void add_in_place(value_type&, value_type, value_type);
    static void subtract_in_place(value_type&, value_type, value_type);
    static void divide_in_place(value_type&, double);
};

template<class Physics>
FVMAssembler<Physics>::FVMAssembler(const Mesh& m, Physics& p)
    : m(m), p(p), is_dirichlet(m.local_nodes()) {
    mpicomm_ = m.mpicomm();
}

template<class Physics>
const mesh::Mesh& FVMAssembler<Physics>::mesh() const {
    return m;
}

template<class Physics>
Physics& FVMAssembler<Physics>::physics() {
    return p;
}

template<class Physics>
template<typename Iterator>
int FVMAssembler<Physics>::compute_residual(
    double time, const_iterator u, const_iterator up, Iterator res) {

    // Preprocess
    physics().preprocess_evaluation(time, mesh(), u, up);

    // Zero the right hand side
    for (int i = 0; i < mesh().local_nodes(); ++i)
        res[i] = value_type();

    // Determine Dirichlet nodes
    for (int i = 0; i < mesh().local_nodes(); ++i) {
        // Given that some variables can be strictly algebraic everywhere on the domain
        // we have to check every node, not just those that lie on the boundary
        is_dirichlet[i] = physics().dirichlet(time, mesh().node(i));
    }

    {// Fluxes
        int i = 0;

        // Interior Flux
        for (; i < mesh().interior_cvfaces(); ++i) {
            const mesh::CVFace& cvf = mesh().cvface(i);
            value_type flux = physics().flux(time, cvf, u);

            int front_id = cvf.front().id();
            if (front_id < mesh().local_nodes())
                add_in_place(res[front_id], is_dirichlet[front_id], flux);
            int back_id = cvf.back().id();

            if (back_id < mesh().local_nodes())
                subtract_in_place(res[back_id], is_dirichlet[back_id], flux);
        }

        // Boundary flux
        for (; i < mesh().cvfaces(); ++i) {
            const mesh::CVFace& cvf = mesh().cvface(i);
            int back_id = cvf.back().id();
            if (back_id < mesh().local_nodes()) {
                value_type flux = physics().boundary_flux(time, cvf, u);
                subtract_in_place(res[back_id], is_dirichlet[back_id], flux);
            }
        }

    }

    // Volume averaged terms
    for (int i = 0; i < mesh().local_nodes(); ++i) {

        const mesh::Volume& v = mesh().volume(i);

        // Scale by volume
        divide_in_place(res[i], v.vol());

        // Source
        value_type source = physics().source(time, v, u);
        add_in_place(res[i], source);

        // Left hand side
        value_type left = physics().lhs(time, v, u, up);
        subtract_in_place(res[i], left);
    }

    /*
    for(int i=0; i<m.local_nodes(); i++)
        std::cout << res[i].h << " ";
    std::cout << std::endl;
    for(int i=0; i<m.local_nodes(); i++)
        std::cout << res[i].M << " ";
    std::cout << std::endl;
    exit(0);
    */

    // Postprocess
    physics().postprocess_evaluation(time, mesh(), u, up);

    return 0;
}

// Basic arithmetic functions on value_type
template<class Physics>
void FVMAssembler<Physics>::add_in_place(
value_type& destination, value_type source) {
    double* dest = reinterpret_cast<double*>(&destination);
    const double* src = reinterpret_cast<const double*>(&source);
    for (int i = 0; i < variables_per_node; ++i) {
        dest[i] += src[i];
    }
}

template<class Physics>
void FVMAssembler<Physics>::add_in_place(
value_type& destination, value_type is_dirichlet, value_type source) {
    double* dest = reinterpret_cast<double*>(&destination);
    const double* is_dir = reinterpret_cast<const double*>(&is_dirichlet);
    const double* src = reinterpret_cast<const double*>(&source);
    for (int i = 0; i < variables_per_node; ++i) {
        if (!is_dir[i]) {
            dest[i] += src[i];
        }
    }
}

template<class Physics>
void FVMAssembler<Physics>::subtract_in_place(
value_type& destination, value_type source) {
    double* dest = reinterpret_cast<double*>(&destination);
    const double* src = reinterpret_cast<const double*>(&source);
    for (int i = 0; i < variables_per_node; ++i) {
        dest[i] -= src[i];
    }
}

template<class Physics>
void FVMAssembler<Physics>::subtract_in_place(
value_type& destination, value_type is_dirichlet, value_type source) {
    double* dest = reinterpret_cast<double*>(&destination);
    const double* is_dir = reinterpret_cast<const double*>(&is_dirichlet);
    const double* src = reinterpret_cast<const double*>(&source);
    for (int i = 0; i < variables_per_node; ++i) {
        if (!is_dir[i]) {
            dest[i] -= src[i];
        }
    }
}

template<class Physics>
void FVMAssembler<Physics>::divide_in_place(
value_type& destination, double val) {
    double* d = reinterpret_cast<double*>(&destination);
    for (int i = 0; i < variables_per_node; ++i) {
        d[i] /= val;
    }
}

// Definition of static member
template<typename ValueType>
const int FVMAssembler<ValueType>::variables_per_node;

} // end namespace fvm

#endif
