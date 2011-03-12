#ifndef SOLVER_H
#define SOLVER_H

#include <fvm/fvm.h>
#include <fvm/mesh.h>
#include <fvm/impl/assemblers/fvm_compact_assembler.h>
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
    typedef typename Physics::TVec TVec;
    typedef typename Physics::TVecDevice TVecDevice;
    typedef typename Physics::TVecDevice::coordinator_type CoordDevice;

    // Constructor
    SolverBase(const Mesh& m, Physics& p, double t0);

    // Returns the current solution time
    double time() const;

    // Returns a reference to the mesh
    const Mesh& mesh() const;

    // Returns a reference to the physics
    Physics& physics() const;

    // returns a reference to the solution vector
    const TVecDevice& solution() const;

private:
    SolverBase(const SolverBase&);
    SolverBase& operator=(const SolverBase&);

protected:
    const mesh::Mesh& m;
    mpi::MPICommPtr mpicomm_;
    mpi::Communicator<CoordDevice, value_type> node_comm_;
    int u_comm_tag_, up_comm_tag_;
    Physics& p;
    double t;
    // DEVICE
    // use minlin to store these vectors
    TVecDevice u;
    TVecDevice up;
    TVecDevice temp;

    // DEVICE
    // this wants to point to a minlin vector
    int compute_residual(TVecDevice &y, bool communicate);
    friend class Callback<Physics>;
};

template<class Physics, class Integrator>
class Solver : public SolverBase<Physics> {
    typedef typename Physics::TVecDevice TVecDevice;
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

    temp = TVecDevice( m.local_nodes()*value_type::variables );
    u = TVecDevice( m.nodes()*value_type::variables );
    up = TVecDevice( m.nodes()*value_type::variables );

    u_comm_tag_ = node_comm_.vec_add(u);
    up_comm_tag_ = node_comm_.vec_add(up);

    physics().initialise(
        t, mesh(),
        u, up, temp,
        Callback<Physics>(this)
    );
}

template<class Physics, class Integrator>
Solver<Physics, Integrator>::Solver(const Mesh& m, Physics& p, Integrator& I, double t0=0.)
    : SolverBase<Physics>(m, p, t0), i(I)
{
    integrator().initialise(
        Base::t,
        Base::u, Base::up,
        Callback<Physics>(this)
    );
}

template<class Physics>
int SolverBase<Physics>::compute_residual(TVecDevice &res, bool communicate) {
    util::Timer timer;
    if (communicate) {
        node_comm_.send(u_comm_tag_);
        node_comm_.send(up_comm_tag_);
        node_comm_.recv_all();
    }

    int retval = Assembler::compute_residual( t, u, up, res );
    return retval;
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
const typename Physics::TVecDevice&
SolverBase<Physics>::solution() const {
    return u;
}

template<class Physics>
int Callback<Physics>::operator()(TVecDevice &y, bool communicate) {
    assert(solver);
    return solver->compute_residual(y, communicate);
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
