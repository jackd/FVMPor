#ifndef BROADCAST_COMMUNICATOR_H
#define BROADCAST_COMMUNICATOR_H

#include <fvm/fvm.h>
#include <fvm/mesh.h>
#include <mpi/mpicomm.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace fvm {

template<typename ValueType>
class BroadcastCommunicator {
public:
    typedef mesh::Mesh Mesh;

    typedef ValueType value_type;
    typedef typename Iterator<value_type>::type iterator;
    typedef typename ConstIterator<value_type>::type const_iterator;

    BroadcastCommunicator(const Mesh& m);
    const Mesh& mesh() const;
    void communicate(iterator values);

private:
    BroadcastCommunicator(const BroadcastCommunicator&);
    BroadcastCommunicator& operator=(const BroadcastCommunicator&);
    
    static const int variables_per_node = VariableTraits<value_type>::number;

    const Mesh& m;
#ifdef NEWCOMM
    mpi::MPICommPtr procinfo;
#endif

    // The mechanism for communicating the values and derivatives at external
    // nodes is rather crude.  We simply do an MPI gather all, so that each
    // process ends up with the values at all nodes.  It can then pull out the
    // ones it needs.
    std::vector<value_type> global_values;

    // Local node counts and vtxdist scaled by variables_per_node
    std::vector<int> scaled_counts;
    std::vector<int> scaled_vtxdist;

    // Wraps a call to MPI_Allgatherv.
    void gather_all(value_type* dst, const value_type* src, int size) const;
};

template<typename ValueType>
const mesh::Mesh& BroadcastCommunicator<ValueType>::mesh() const {
    return m;
}

template<typename ValueType>
BroadcastCommunicator<ValueType>::BroadcastCommunicator(const Mesh& m)
    : m(m), global_values(m.global_nodes())
{
#ifdef NEWCOMM
    //procinfo = boost::const_pointer_cast<mpi::MPIComm>(m.mpicomm());
    procinfo = m.mpicomm()->duplicate("BCast");
#else
    mpi::MPIComm procinfo;
#endif

    std::vector<int> counts;
    const std::vector<int>& vtxdist = mesh().vtxdist();
    std::adjacent_difference(
        vtxdist.begin(), vtxdist.end(), std::back_inserter(counts));
#ifdef NEWCOMM
    assert(int(counts.size()) == procinfo->size()+1);
#else
    assert(int(counts.size()) == procinfo.size()+1);
#endif

    scaled_counts.resize(counts.size());
    std::transform(
        counts.begin(), counts.end(), scaled_counts.begin(),
        std::bind2nd(std::multiplies<int>(), variables_per_node)
    );

    scaled_vtxdist.resize(counts.size());
    std::transform(
        mesh().vtxdist().begin(), mesh().vtxdist().end(), scaled_vtxdist.begin(),
        std::bind2nd(std::multiplies<int>(), variables_per_node)
    );
}

template<typename ValueType>
void BroadcastCommunicator<ValueType>::communicate(iterator values)
{
    // gather the entire global vector
    gather_all(&global_values.front(), &values[0], mesh().local_nodes());

    // copy the external values
    for (int i = 0; i < mesh().external_nodes(); ++i)
        values[mesh().local_nodes() + i] = global_values[mesh().external_node_id(i)];
}

template<typename ValueType>
void BroadcastCommunicator<ValueType>::
gather_all(value_type* dst, const value_type* src, int size) const
{
#ifndef NEWCOMM
    mpi::MPIComm procinfo;
#endif
    MPI_Allgatherv(
        const_cast<value_type*>(src),
        size * variables_per_node,
        MPI_DOUBLE,
        dst,
        const_cast<int*>(&scaled_counts[1]),
        const_cast<int*>(&scaled_vtxdist[0]),
        MPI_DOUBLE,
#ifdef NEWCOMM
        procinfo->communicator()
#else
        procinfo.communicator()
#endif
    );
}

// Definition of static member
template<typename ValueType>
const int BroadcastCommunicator<ValueType>::variables_per_node;

} // end namespace fvm

#endif
