#ifndef PATTERN_H
#define PATTERN_H

#include <algorithm>
#include <numeric>
#include <vector>
#include <map>

#include <mpi/mpicomm.h>

namespace mesh {

class Pattern{
    public:
        Pattern() {};
        Pattern(mpi::MPICommPtr);

        // return information about pattern to caller
        const std::vector<int>& neighbour_list() const;
        int num_neighbours() const;
        int  neighbour(int n) const;
        const std::vector<int>& send_index(int n) const;
        const std::vector<int>& recv_index(int n) const;
        mpi::MPICommPtr comm() const;
        bool is_neighbour( int n ) const;

        // add a neighbour to the pattern
        void add_neighbour( int n, std::vector<int>& send, std::vector<int>& recv );

        // sanity check the pattern to ensure that it is consistent over all subdomains.
        bool verify_pattern() const;
    private:
        mpi::MPICommPtr comm_;
        std::vector<int> neighbours_;
        std::map<int,std::vector<int> > send_index_;
        std::map<int,std::vector<int>  > recv_index_;
};

inline
Pattern::Pattern( mpi::MPICommPtr comm ){
    comm_ = comm;
}

inline
const std::vector<int>& Pattern::neighbour_list() const{
    return neighbours_;
}

inline
int Pattern::num_neighbours() const{
    return neighbours_.size();
}

inline
mpi::MPICommPtr Pattern::comm() const{
    return comm_;
}

inline
int  Pattern::neighbour(int n) const{
    assert( n>=0 && n<num_neighbours() );
    return neighbours_[n];
}

inline
const std::vector<int>& Pattern::send_index(int n) const{
    std::map<int,std::vector<int> >::const_iterator it = send_index_.find(n);
    assert( it!=send_index_.end());
    return it->second;
}

inline
const std::vector<int>& Pattern::recv_index(int n) const{
    std::map<int,std::vector<int> >::const_iterator it = recv_index_.find(n);
    assert( it!=recv_index_.end());
    return it->second;
}

inline
bool Pattern::is_neighbour( int n ) const{
    return( std::find(neighbours_.begin(), neighbours_.end(), n) != neighbours_.end() );
}

// add a neighbour to the pattern
inline
void Pattern::add_neighbour( int n, std::vector<int>& send, std::vector<int>& recv ){
    // sanity check
    //assert(n>=0 && n<comm_.size()); // ensure that n is in the MPI group
    assert(n>=0 && n<comm_->size()); // ensure that n is in the MPI group
    assert( !is_neighbour(n) ); // ensure that this neighbour has not already been added

    neighbours_.push_back(n);
    send_index_[n] = send;
    recv_index_[n] = recv;
}

} // namespace fvm

#endif
