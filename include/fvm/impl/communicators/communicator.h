#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <algorithm>
#include <numeric>
#include <vector>
#include <map>

#include <fvm/impl/communicators/pattern.h>
#include <mpi/mpicomm.h>

namespace mpi {

    template<typename T>
    struct block_traits {
        enum {blocksize = T::variables};
    };
    template<>
    struct block_traits<double> {
        enum {blocksize = 1};
    };

    struct MPIDataType{
        MPI_Datatype MPIType;
        void set(MPI_Datatype T) {MPIType = T;};
    };

    // communicator used to coordinate the communication of information
    // stored in arrays that have overlapping implied by a Pattern
    template <typename Type>
    class Communicator{
        typedef std::vector<Type> TypeVector;
        public:
            Communicator( const std::string &str, const mesh::Pattern& );
            Communicator() : pattern_(0) {};
            const mesh::Pattern& pattern() const;
            void set_pattern( const std::string &str, const mesh::Pattern& );

            // add and remove vectors to and from the communicator
            int vec_add(Type*);
            int vec_remove( int );

            MPICommPtr mpicomm() const {return comm_;};
            
            // communication
            int recv( int );
            int send( int );
            int recv_all();
        private:
            int neighbours() const { return pattern().num_neighbours(); };
            int neighbour( int n ) const { return pattern().neighbour(n); };
            Communicator(const Communicator&);
            Communicator& operator=(const Communicator&);
            void build_from_pattern();

            // the pattern that describes how to distribute information to and from neighbours
            const mesh::Pattern* pattern_;

            int block_size_;

            // a list of tags associated with the vectors using the communicator
            std::set<int> vectors_;
            // a map with tag as index which indicates whether a vector is communicating
            std::map<int,bool> busy_;
            // tag as index of pointers to the actual data of each vector
            std::map<int,Type*> data_;

            // the MPI communicator
            MPICommPtr comm_;

            // name of the communicator
            std::string name_;
            
            // describe the indexed data to MPI
            std::map<int,MPIDataType> send_type_;
            std::map<int,MPIDataType> recv_type_;
            std::vector<int> block_lengths_;
            std::map<int,std::vector<int> > send_displacements_;
            std::map<int,std::vector<int> > recv_displacements_;

            // maps with vector tag as key and a vector of MPI_Request for each of sends and receives to neighbours
            std::map<int,std::vector<MPI_Request> > send_requests_;
            std::map<int,std::vector<MPI_Request> > recv_requests_;
    };

    template<typename Type>
    Communicator<Type>::Communicator( const std::string &str, const mesh::Pattern& pat ) 
        : pattern_(&pat)
    {
        name_ = pattern().comm()->name_short()+"_"+str;
        comm_ = pattern().comm()->duplicate(name_);
        build_from_pattern();
    }

    template<typename Type>
    void Communicator<Type>::set_pattern(const std::string &str, const mesh::Pattern& pat)
    {
        pattern_ = &pat;

        name_ = pattern().comm()->name_short()+"_"+str;

        // create an MPI communicator based on that for the Pattern
        comm_ = pattern().comm()->duplicate(name_);
        
        build_from_pattern();
    };

    template<typename Type>
    void Communicator<Type>::build_from_pattern(){
        *comm_ << "Communicator::build_from_pattern" << std::endl;

        // query Type to determine the block size
        block_size_ = block_traits<Type>::blocksize;
        *comm_ << "\tblock_size_ = " << block_size_ << std::endl;

        // create buffers and keys for communication
        // this occurs in a few steps
        // STEP 1: find displacements in the vectors
        size_t maxLen = 0;
        for(int i=0; i<pattern().num_neighbours(); i++)
        {
            int n = pattern().neighbour(i);
            const std::vector<int>& recv_perm = pattern().recv_index(n);
            const std::vector<int>& send_perm = pattern().send_index(n);
            maxLen = std::max(std::max(maxLen, recv_perm.size()), send_perm.size());

            send_displacements_[n].resize(send_perm.size());
            std::transform(send_perm.begin(), send_perm.end(), send_displacements_[n].begin(), std::bind2nd(std::multiplies<int>(), block_size_));
            recv_displacements_[n].resize(recv_perm.size());
            std::transform(recv_perm.begin(), recv_perm.end(), recv_displacements_[n].begin(), std::bind2nd(std::multiplies<int>(), block_size_));
        }
        // STEP 2: make a vector of block sizes, which are always equal for each neighbour
        //         so just allocate one vector large enough for every buffer
        block_lengths_.assign(maxLen, block_size_);

        // STEP 3: create the MPI_Datatype for each send and receive operation to a neighbour [page 96]
        for(int i=0; i<pattern().num_neighbours(); i++)
        {
            MPI_Datatype tmpType;
            int n = pattern().neighbour(i);

            int flag = MPI_Type_indexed( send_displacements_[n].size(),
                                         const_cast<int*>(&block_lengths_[0]),
                                         const_cast<int*>(&send_displacements_[n][0]),
                                         MPI_DOUBLE,
                                         &tmpType );
            assert( flag==MPI_SUCCESS );
            send_type_[n].set(tmpType);
            assert( MPI_Type_commit(reinterpret_cast<MPI_Datatype*>(&send_type_[n]))==MPI_SUCCESS );
            
            flag = MPI_Type_indexed(     recv_displacements_[n].size(),
                                         const_cast<int*>(&block_lengths_[0]),
                                         const_cast<int*>(&recv_displacements_[n][0]),
                                         MPI_DOUBLE, &tmpType );
            assert( flag==MPI_SUCCESS );
            recv_type_[n].set(tmpType);
            assert( MPI_Type_commit(reinterpret_cast<MPI_Datatype*>(&recv_type_[n]))==MPI_SUCCESS );
            *comm_ << "\tsend type to neigbour " << n << " is " << send_type_[n].MPIType << std::endl;
            *comm_ << "\trecv type to neigbour " << n << " is " << recv_type_[n].MPIType << std::endl;
        }
    } 


    // add a vector to the communicator
    // the vector must keep the tag that is returned to identify itself
    // for later interactions with the communicator
    template<typename Type>
    int Communicator<Type>::vec_add(Type *data){
        // determine new tag for the vector
        int new_tag = 1;
        if( vectors_.size() )
            new_tag = *std::max_element(vectors_.begin(), vectors_.end()) + 1;
        *comm_ << "adding vector with tag " << new_tag << std::endl;
        vectors_.insert(new_tag);

        send_requests_[new_tag].resize( neighbours() );
        recv_requests_[new_tag].resize( neighbours() );
        busy_[new_tag] = false;
        data_[new_tag] = data;

        return new_tag;
    }

    // remove a vector from the communicator
    template<typename Type>
    int Communicator<Type>::vec_remove( int tag ){
        // WARNING
        // Need to clean up any unfinished communication for the vector associated with tag

        // remove the vector tag from the tag list
        std::set<int>::const_iterator it = vectors_.find(tag);
        *comm_ << "removing vector with tag " << tag << std::endl;
        assert( it!=vectors_.end() );
        vectors_.erase(it);

        // remove associated information
        send_requests_.erase(tag);
        recv_requests_.erase(tag);
        busy_.erase(tag);
        data_.erase(tag);

        return tag;
    }

    template<typename Type>
    int Communicator<Type>::send( int tag ){
        // check that we have been asked for a valid tag
        assert( std::find(vectors_.begin(), vectors_.end(), tag)!=vectors_.end() );

        // assert that we are not requesting a send on a vector that has a pending receive 
        assert( !busy_[tag] );

        // make the send
        for( int i=0; i<neighbours(); i++ ){
            int n = pattern().neighbour(i);

            send_requests_[tag][i] = comm_->Isend( data_[tag], n, tag, send_type_[n].MPIType );
            recv_requests_[tag][i] = comm_->Irecv( data_[tag], n, tag, recv_type_[n].MPIType );
        }
        busy_[tag] = true;
        
        return tag;
    }

    // complete pending communication associated with vector with tag
    template<typename Type>
    int Communicator<Type>::recv( int tag ){
        // check that we have been asked for a valid tag
        assert( std::find(vectors_.begin(), vectors_.end(), tag)!=vectors_.end() );

        // assert that we are not requesting a recv on a vector that has no pending recv 
        assert( busy_[tag] );

        int send_size = send_requests_[tag].size();
        int recv_size = recv_requests_[tag].size();
        std::vector<MPI_Status> tmpStatus;
        tmpStatus.resize(std::max(send_size, recv_size));

        comm_->Waitall(send_requests_[tag], tmpStatus);
        comm_->Waitall(recv_requests_[tag], tmpStatus);

        busy_[tag] = false;

        return tag;
    }

    // complete all pending communication associated with vectors
    template<typename Type>
    int Communicator<Type>::recv_all(){
        for( std::set<int>::iterator i=vectors_.begin(); i!=vectors_.end(); i++ ){
            int tag = *i;
            if( busy_[tag] ){
                recv(tag);
            }
        }
        return 0;
    }

    // return a reference to the Pattern
    template<typename Type>
    const mesh::Pattern& Communicator<Type>::pattern() const {
        assert(pattern_);
        return *pattern_;
    }

}// namespace fvm

#endif
