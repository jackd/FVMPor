/**************************************************************************
 * The compact assembler hands construction of the residual to the physics.
 * This allows the implementer of the physics to implement a more efficient
 * assembler than the default assembler, which can be orders of magnitude
 * faster for meshes with many control volume faces
 **************************************************************************/
#ifndef FVM_ASSEMBLER_H
#define FVM_ASSEMBLER_H

#include <fvm/fvm.h>
#include <fvm/mesh.h>
#include <mpi/mpicomm.h>
#include <util/timer.h>

#include <vector>

namespace fvm {

template<class Physics>
class FVMAssembler {
public:
    typedef mesh::Mesh Mesh;

    typedef typename Physics::value_type value_type;
    typedef typename Physics::TVecDevice TVecDevice;

    FVMAssembler(const Mesh& m, Physics& p);
    const Mesh& mesh() const;
    Physics& physics();
    int compute_residual( double time,
                          const TVecDevice &u,
                          const TVecDevice &up,
                          TVecDevice &res       );
private:
    FVMAssembler(const FVMAssembler&);
    FVMAssembler& operator=(const FVMAssembler&);

    const Mesh& m;
    Physics& p;
    mpi::MPICommPtr mpicomm_;


    static const int variables_per_node = VariableTraits<value_type>::number;
};

template<class Physics>
FVMAssembler<Physics>::FVMAssembler(const Mesh& m, Physics& p)
    : m(m), p(p) {
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
int FVMAssembler<Physics>::compute_residual(
    double time, const TVecDevice &u, const TVecDevice &up, TVecDevice &res) {
    //std::cerr << "==========================" << std::endl;
    //std::cerr << "Assembler residual compute" << std::endl;
    //std::cerr << "==========================" << std::endl;

    // Preprocess
    physics().preprocess_evaluation(time, mesh(), u, up);

    // find the residual
    physics().residual_evaluation(time, mesh(), u, up, res);

    //for(int i=0; i<mesh().local_nodes(); i++)
    //    std::cerr << res[i] << ' ';
    //std::cerr << std::endl;
    
    // Postprocess
    //physics().postprocess_evaluation(time, mesh(), u, up);

    return 0;
}

// Definition of static member
template<typename ValueType>
const int FVMAssembler<ValueType>::variables_per_node;

} // end namespace fvm

#endif
