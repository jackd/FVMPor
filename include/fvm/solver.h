#ifndef SOLVER_H
#define SOLVER_H

#include <fvm/fvm.h>
#include <fvm/mesh.h>
#include <fvm/impl/assemblers/fvm_assembler.h>
#include <fvm/impl/communicators/communicator.h>

#include <vector>

namespace fvm {

template<class Physics>
class SolverBase :
    private FVMAssembler<Physics>{

    typedef FVMAssembler<Physics> Assembler;

public:
    typedef mesh::Mesh Mesh;

    typedef typename Physics::value_type value_type;
    typedef typename Iterator<value_type>::type iterator;
    typedef typename ConstIterator<value_type>::type const_iterator;

    // Constructor
    SolverBase(const Mesh& m, Physics& p, double t0);

    // Returns the current solution time
    double time() const;

    // Returns a reference to the mesh
    const Mesh& mesh() const;

    // Returns a reference to the physics
    Physics& physics() const;

    // Returns iterators that designate the value at the current time
    const_iterator begin() const; // local nodes
    const_iterator end() const; // local nodes
    const_iterator end_ext() const; // local and external nodes

    // Returns iterators that designate the derivative at the current time
    const_iterator pbegin() const;// local nodes
    const_iterator pend() const;// local nodes
    const_iterator pend_ext() const;// local and external nodes

private:
    SolverBase(const SolverBase&);
    SolverBase& operator=(const SolverBase&);

protected:
    const mesh::Mesh& m;
    mpi::MPICommPtr mpicomm_;
    mpi::Communicator<value_type> node_comm_;
    int u_comm_tag_, up_comm_tag_;
    Physics& p;
    double t;
    std::vector<value_type> u;
    std::vector<value_type> up;
    std::vector<value_type> temp;

    template<typename Iterator>
    int compute_residual(Iterator it, bool communicate);
    friend class Callback<Physics>;
};

template<class Physics, class Integrator>
class Solver : public SolverBase<Physics> {
public:
    typedef SolverBase<Physics> Base;
    typedef mesh::Mesh Mesh;

    // Constructor
    Solver(const Mesh& m, Physics& p, Integrator& i, double tt);

    // Advances solution in time
    void advance();                  // by one internal timestep
    void advance(double next_time);  // to specified time

    // Returns a reference to the integrator
    Integrator& integrator() const;

private:
    Solver(const Solver&);
    Solver& operator=(const Solver&);
    Integrator& i;
};

template<class Physics>
SolverBase<Physics>::SolverBase(const Mesh& m, Physics& p, double t0)
    : Assembler(m, p),
      m(m), p(p),
      t(t0)
{
    mpicomm_ = m.mpicomm();
    node_comm_.set_pattern( "NP_Type", m.node_pattern() );

    temp.resize(m.local_nodes());
    up.resize(m.nodes());
    u.resize(m.nodes());
    u_comm_tag_ = node_comm_.vec_add(&u[0]);
    up_comm_tag_ = node_comm_.vec_add(&up[0]);
    physics().initialise(
        t, mesh(),
        make_iterator<value_type>(&u[0], &u[0], &u[0] + mesh().local_nodes()),
        make_iterator<value_type>(&up[0], &up[0], &up[0] + mesh().local_nodes()),
        make_iterator<value_type>(&temp[0], &temp[0], &temp[0] + mesh().local_nodes()),
        Callback<Physics>(this)
    );
}

template<class Physics, class Integrator>
Solver<Physics, Integrator>::Solver(const Mesh& m, Physics& p, Integrator& I, double t0=0.)
    : SolverBase<Physics>(m, p, t0), i(I)
{
    integrator().initialise(
        Base::t,
        make_iterator<typename Base::value_type>(
            &Base::u[0], &Base::u[0], &Base::u[0] + Base::mesh().nodes()),
        make_iterator<typename Base::value_type>(
            &Base::up[0], &Base::up[0], &Base::up[0] + Base::mesh().nodes()),
        Callback<Physics>(this)
    );
}

template<class Physics>
template<typename Iterator>
int SolverBase<Physics>::compute_residual(Iterator res, bool communicate) {
    if (communicate) {
        node_comm_.send(u_comm_tag_);
        node_comm_.send(up_comm_tag_);
        node_comm_.recv_all();
    }
    return Assembler::compute_residual(
        t,
        make_iterator<const value_type>(&u[0], &u[0], &u[0] + mesh().nodes()),
        make_iterator<const value_type>(&up[0], &up[0], &up[0] + mesh().nodes()),
        res
    );
}

template<class Physics>
double SolverBase<Physics>::time() const {
    return t;
}

template<class Physics>
const mesh::Mesh& SolverBase<Physics>::mesh() const {
    return m;
}

template<class Physics>
Physics& SolverBase<Physics>::physics() const {
    return p;
}

template<class Physics, class Integrator>
Integrator& Solver<Physics, Integrator>::integrator() const {
    return i;
}

template<class Physics>
typename SolverBase<Physics>::const_iterator
SolverBase<Physics>::end_ext() const {
    return make_iterator<const value_type>(
        &u[0] + mesh().nodes(), &u[0], &u[0] + mesh().nodes());
}

template<class Physics>
typename SolverBase<Physics>::const_iterator
SolverBase<Physics>::begin() const {
    return make_iterator<const value_type>(
        &u[0], &u[0], &u[0] + mesh().local_nodes());
}

template<class Physics>
typename SolverBase<Physics>::const_iterator
SolverBase<Physics>::end() const {
    return make_iterator<const value_type>(
        &u[0] + mesh().local_nodes(), &u[0], &u[0] + mesh().local_nodes());
}

template<class Physics>
typename SolverBase<Physics>::const_iterator
SolverBase<Physics>::pbegin() const {
    return make_iterator<const value_type>(
        &up[0], &up[0], &up[0] + mesh().local_nodes());
}

template<class Physics>
typename SolverBase<Physics>::const_iterator
SolverBase<Physics>::pend() const {
    return make_iterator<const value_type>(
        &up[0] + mesh().local_nodes(), &up[0], &up[0] + mesh().local_nodes());
}

template<class Physics>
typename SolverBase<Physics>::const_iterator
SolverBase<Physics>::pend_ext() const {
    return make_iterator<const value_type>(
        &up[0] + mesh().nodes(), &up[0], &up[0] + mesh().nodes());
}

template<class Physics>
template<typename Iterator>
int Callback<Physics>::operator()(Iterator it, bool communicate) {
    assert(solver);
    return solver->compute_residual(it, communicate);
}

template<class Physics, class Integrator>
void Solver<Physics, Integrator>::advance() {
    integrator().advance();

    Base::node_comm_.send(Base::u_comm_tag_);
    Base::node_comm_.recv(Base::u_comm_tag_);
}

template<class Physics, class Integrator>
void Solver<Physics, Integrator>::advance(double next_time) {
    integrator().advance(next_time);

    Base::node_comm_.send(Base::u_comm_tag_);
    Base::node_comm_.recv(Base::u_comm_tag_);
}

} // end namespace fvm

#endif
