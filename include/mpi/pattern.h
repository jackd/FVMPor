#ifndef PATTERN_H
#define PATTERN_H

#include <algorithm>
#include <numeric>
#include <vector>
#include <map>

#include "mpicomm.h"

namespace mpi {
    class Pattern{
        typedef std::vector<int> intVec;
        public:
            Pattern(MPIComm&);

            // return information about pattern to caller
            const intVec& neightbours() const;
            int  neightbour(int n) const;
            const intVec& send_index(int n) const;
            const intVec& recv_index(int n) const;
            const MPIComm& comm() const;

            // add a neighbour to the pattern
            void add_neighbour( int n, intVec& send, intVec& recv );

            // sanity check the pattern to ensure that it is consistent over all subdomains.
            bool verify_pattern() const;
        private:
            MPIComm comm_;
            intVec neigbours_;
            std::map<int,intVec> send_index_;
            std::map<int,intVec> recv_index_;
    }

    Pattern::Pattern( MPI_Comm& ){
        comm = MPI_Comm;
    }

    const intVec& Pattern::neighbours() const{
        return neighbours_.size();
    }

    MPIComm& Pattern::comm() const{
        return &comm_;
    }

    int  Pattern::neightbour(int n) const{
        assert( n>=0 && n<neighbours() )
        return neigbours_[n];
    }

    const intVec& Pattern::send_index(int n) const{
        std::map<int,intVec>::const_iterator it = send_index_.find(tag);
        assert( it!=send_index_.end());
        return it->second;
    }

    const intVec& Pattern::recv_index(int n) const{
        std::map<int,intVec>::const_iterator it = recv_index_.find(tag);
        assert( it!=recv_index_.end());
        return it->second;
    }

    bool Pattern::is_neighbour( int n ) const{
        return( std::find(neighbours_.begin(), neighbours_.end(), n) != neightbours_.end() );
    }

    // add a neighbour to the pattern
    Pattern::add_neighbour( int n, intVec& send, intVec& recv ){
        // sanity check
        assert(n>=0 && n<comm_.rank()); // ensure that n is in the MPI group
        assert( !is_neighbour(n) ); // ensure that this neighbour has not already been added

        neighbours_.push_back(n);
        send_index_[n] = send;
        recv_index_[n] = recv;
    }
} // namespace fvm

#endif
