#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <algorithm>
#include <numeric>
#include <vector>
#include <map>

#include <fvm/impl/communicators/pattern.h>
#include <mpi/mpicomm.h>

#include <lin/lin.h>
#include <lin/impl/rebind.h>

#include <util/coordinators.h>
#include <util/timer.h>
#include <util/timer_cuda.h>

namespace mpi {

    using util::CoordTraits;

    template<typename T>
    struct block_traits {
        enum {blocksize = T::variables};
        typedef double baseT;
        static const MPI_Datatype MPI_type = MPI_DOUBLE;
    };
    template<>
    struct block_traits<double> {
        enum {blocksize = 1};
        typedef double baseT;
        static const MPI_Datatype MPI_type = MPI_DOUBLE;
    };
    template<>
    struct block_traits<int> {
        enum {blocksize = 1};
        typedef int baseT;
        static const MPI_Datatype MPI_type = MPI_INT;
    };

    struct MPIDataType{
        MPI_Datatype MPIType;
        void set(MPI_Datatype T) {MPIType = T;};
    };

    // communicator used to coordinate the communication of information
    // stored in arrays that have overlapping implied by a Pattern
    template <typename Coord, typename Type>
    class Communicator{
        typedef std::vector<Type> TypeVector;
        typedef typename block_traits<Type>::baseT baseT;
        typedef typename lin::rebind<Coord, baseT>::type CoordT;
        typedef typename lin::rebind<Coord, int>::type CoordIndex;
        typedef lin::Vector<baseT, CoordT> TVec;
        typedef lin::Vector<int,  CoordIndex> TVecIndex;
        typedef lin::Vector<baseT, lin::DefaultCoordinator<baseT> > TVecHost;
        typedef lin::Vector<int, lin::DefaultCoordinator<int> > TVecHostIndex;
        public:
            Communicator( const std::string &str, const mesh::Pattern& );
            Communicator() : pattern_(0) {};
            const mesh::Pattern& pattern() const;
            void set_pattern( const std::string &str, const mesh::Pattern& );

            // add and remove vectors to and from the communicator
            int vec_add(TVec&);
            int vec_remove(int);

            MPICommPtr mpicomm() const {return comm_;};
            
            // communication
            int recv( int );
            int send( int );
            int recv_all();
        private:
            MPI_Datatype MPI_base_T;
            int neighbours() const { return pattern().num_neighbours(); };
            int neighbour( int n ) const { return pattern().neighbour(n); };
            Communicator(const Communicator&);
            Communicator& operator=(const Communicator&);
            void build_from_pattern();
            void build_on_host_from_pattern();
            void build_on_device_from_pattern();

            // the pattern that describes how to distribute
            // information to and from neighbours
            const mesh::Pattern* pattern_;

            int block_size_;
            bool data_on_host_;

            // a list of tags associated with the vectors using the communicator
            std::set<int> vectors_;
            // a map with tag as index which indicates whether a
            // vector is communicating
            std::map<int,bool> busy_;

            // buffers used to buffer communication when data
            // is stored on a device
            TVec device_buffer_;
            std::map<int,TVecHost> send_buffer_;
            std::map<int,TVecHost> recv_buffer_;
            std::map<int,TVec> vector_;
            std::map<int,int> send_offset_;
            std::map<int,int> recv_offset_;
            TVecIndex send_perm_;
            TVecIndex recv_perm_;

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

    template<typename Coord, typename Type>
    Communicator<Coord, Type>::Communicator( const std::string &str, const mesh::Pattern& pat ) 
        : pattern_(&pat)
    {
        name_ = pattern().comm()->name_short()+"_"+str;
        comm_ = pattern().comm()->duplicate(name_);

        MPI_base_T = block_traits<Type>::MPI_type;

        build_from_pattern();
    }

    template<typename Coord, typename Type>
    void Communicator<Coord, Type>::set_pattern(const std::string &str, const mesh::Pattern& pat)
    {
        pattern_ = &pat;

        name_ = pattern().comm()->name_short()+"_"+str;
        MPI_base_T = block_traits<Type>::MPI_type;

        // create an MPI communicator based on that for the Pattern
        comm_ = pattern().comm()->duplicate(name_);

        build_from_pattern();
    };

    template<typename Coord, typename Type>
    void Communicator<Coord, Type>::build_from_pattern(){
        data_on_host_ = !CoordTraits<Coord>::is_device();
        if(data_on_host_)
            build_on_host_from_pattern();
        else
            build_on_device_from_pattern();
    }

    template<typename Coord, typename Type>
    void Communicator<Coord, Type>::build_on_device_from_pattern(){
        *comm_ << "Communicator::build_on_device_from_pattern" << std::endl;

        // query Type to determine the block size
        block_size_ = block_traits<Type>::blocksize;
        *comm_ << "\tblock_size_ = " << block_size_ << std::endl;

        // determine the displacements in the send and receive buffers
        // for making the MPI Types for each neighbour
        int n_recv = 0;
        int n_send = 0;
        for(int i=0; i<pattern().num_neighbours(); i++)
        {
            int n = pattern().neighbour(i);
            size_t to_send = pattern().send_index(n).size();
            size_t to_recv = pattern().recv_index(n).size();

            send_offset_[n] = n_send;
            recv_offset_[n] = n_recv;

            n_send += to_send*block_size_;
            n_recv += to_recv*block_size_;
        }

        // make the permutation vectors used for packing/unpacking the
        // data being communicated
        TVecHostIndex send_perm(n_send);
        TVecHostIndex recv_perm(n_recv);
        int recv_idx=0, send_idx=0;
        for(int i=0; i<pattern().num_neighbours(); i++)
        {
            int n = pattern().neighbour(i);
            const std::vector<int>& send_patt = pattern().send_index(n);
            const std::vector<int>& recv_patt = pattern().recv_index(n);

            for(int j=0; j<send_patt.size(); j++, send_idx+=block_size_)
                for(int k=0; k<block_size_; k++)
                    send_perm[send_idx+k] = send_patt[j]*block_size_+k;

            for(int j=0; j<recv_patt.size(); j++, recv_idx+=block_size_)
                for(int k=0; k<block_size_; k++)
                    recv_perm[recv_idx+k] = recv_patt[j]*block_size_+k;
        }

        // copy these permutations to the device
        send_perm_ = send_perm;
        recv_perm_ = recv_perm;

        // vector used for temporary copying to/from the device
        // we use the same buffer for both sends and receives
        // so allocate enough room to accomodate both operations
        device_buffer_ = TVec(std::max(n_recv, n_send));

        // make a vector of block sizes, which are always equal for
        // both sends and receives, so allocate one vector large enough for
        // both cases

        // create the MPI_Datatype for each send and receive operation
        for(int i=0; i<pattern().num_neighbours(); i++)
        {
            MPI_Datatype tmpType;
            int n = pattern().neighbour(i);

            int flag = MPI_Type_contiguous(
                            pattern().send_index(n).size()*block_size_,
                            MPI_base_T,
                            &tmpType );
            assert( flag==MPI_SUCCESS );
            send_type_[n].set(tmpType);
            assert( MPI_Type_commit(reinterpret_cast<MPI_Datatype*>(&send_type_[n]))
                      == MPI_SUCCESS );

            flag = MPI_Type_contiguous(
                            pattern().recv_index(n).size()*block_size_,
                            MPI_base_T,
                            &tmpType );
            assert( flag==MPI_SUCCESS );
            recv_type_[n].set(tmpType);
            assert( MPI_Type_commit(reinterpret_cast<MPI_Datatype*>(&recv_type_[n]))
                      == MPI_SUCCESS );
            *comm_ << "\tsend type to neigbour " << n << " is "
                   << send_type_[n].MPIType << std::endl;
            *comm_ << "\trecv type to neigbour " << n << " is "
                   << recv_type_[n].MPIType << std::endl;
        }
    } 

    template<typename Coord, typename Type>
    void Communicator<Coord, Type>::build_on_host_from_pattern(){
        *comm_ << "Communicator::build_from_pattern" << std::endl;

        // query Type to determine the block size
        block_size_ = block_traits<Type>::blocksize;
        *comm_ << "\tblock_size_ = " << block_size_ << std::endl;

        *comm_ << "\tdata_on_host_ = " << (data_on_host_ ? std::string("true") : std::string("false") ) << std::endl;

        // create buffers and keys for communication
        // this occurs in a few steps
        // STEP 1: find displacements in the vectors
        size_t maxLen = 0;
        for(int i=0; i<pattern().num_neighbours(); i++)
        {
            int n = pattern().neighbour(i);
            const std::vector<int>& recv_perm = pattern().recv_index(n);
            const std::vector<int>& send_perm = pattern().send_index(n);
            maxLen = std::max(  std::max(maxLen, recv_perm.size()),
                                send_perm.size() );

            send_displacements_[n].resize(send_perm.size());
            std::transform( send_perm.begin(),
                            send_perm.end(),
                            send_displacements_[n].begin(),
                            std::bind2nd( std::multiplies<int>(),
                                          block_size_) );
            recv_displacements_[n].resize(recv_perm.size());
            std::transform( recv_perm.begin(),
                            recv_perm.end(),
                            recv_displacements_[n].begin(),
                            std::bind2nd( std::multiplies<int>(),
                                          block_size_) );
        }
        // vector used for temporary copying to/from the device
        if( !data_on_host_ )
            device_buffer_ = TVec(maxLen);

        // STEP 2: make a vector of block sizes, which are always equal for
        //         each neighbour so just allocate one vector large enough for
        //         every buffer
        block_lengths_.assign(maxLen, block_size_);

        // STEP 3: create the MPI_Datatype for each send and receive operation
        // to a neighbour [page 96]
        for(int i=0; i<pattern().num_neighbours(); i++)
        {
            MPI_Datatype tmpType;
            int n = pattern().neighbour(i);

            int flag = MPI_Type_indexed( send_displacements_[n].size(),
                                         const_cast<int*>(&block_lengths_[0]),
                                         const_cast<int*>(&send_displacements_[n][0]),
                                         MPI_base_T,
                                         &tmpType );
            assert( flag==MPI_SUCCESS );
            send_type_[n].set(tmpType);
            assert( MPI_Type_commit(reinterpret_cast<MPI_Datatype*>(&send_type_[n]))
                      == MPI_SUCCESS );
            
            flag = MPI_Type_indexed(     recv_displacements_[n].size(),
                                         const_cast<int*>(&block_lengths_[0]),
                                         const_cast<int*>(&recv_displacements_[n][0]),
                                         MPI_base_T, &tmpType );
            assert( flag==MPI_SUCCESS );
            recv_type_[n].set(tmpType);
            assert( MPI_Type_commit(reinterpret_cast<MPI_Datatype*>(&recv_type_[n]))
                      == MPI_SUCCESS );
            *comm_ << "\tsend type to neigbour " << n << " is "
                   << send_type_[n].MPIType << std::endl;
            *comm_ << "\trecv type to neigbour " << n << " is "
                   << recv_type_[n].MPIType << std::endl;
        }
    } 


    // add a vector to the communicator
    // the vector must keep the tag that is returned to identify itself
    // for later interactions with the communicator
    template<typename Coord, typename Type>
    int Communicator<Coord, Type>::vec_add(TVec &v){
        // determine new tag for the vector
        int new_tag = 1;
        if( vectors_.size() )
            new_tag = *std::max_element(vectors_.begin(), vectors_.end()) + 1;
        *comm_ << "adding vector with tag " << new_tag << std::endl;
        vectors_.insert(new_tag);

        send_requests_[new_tag].resize( neighbours() );
        recv_requests_[new_tag].resize( neighbours() );
        busy_[new_tag] = false;

        // communication buffer is always on the host. If the base vector
        // type is on the host, then make the buffer point directly to
        // the vector, otherwise create a new host vector that will
        // be copied into/out of before/after each send/receive.
        if( data_on_host_ ){
            recv_buffer_[new_tag] = TVecHost(v.size(), v.data());
            send_buffer_[new_tag] = TVecHost(v.size(), v.data());
        }
        else{
            recv_buffer_[new_tag] = TVecHost(recv_perm_.size());
            send_buffer_[new_tag] = TVecHost(send_perm_.size());
        }

        // keep a reference to the original data
        vector_[new_tag] = TVec(v.size(), v.data());

        return new_tag;
    }

    // remove a vector from the communicator
    template<typename Coord, typename Type>
    int Communicator<Coord, Type>::vec_remove( int tag ){
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
        send_buffer_.erase(tag);
        recv_buffer_.erase(tag);
        vector_.erase(tag);

        return tag;
    }

    template<typename Coord, typename Type>
    int Communicator<Coord, Type>::send( int tag ){
        //*comm_ << "Vector " << tag << " : initiating sends" << std::endl;
        // check that we have been asked for a valid tag
        assert( std::find(vectors_.begin(), vectors_.end(), tag)!=vectors_.end() );

        // assert that we are not requesting a send on a vector that has a pending receive 
        assert( !busy_[tag] );

        if( pattern().num_neighbours() ){
            // if the data is on the device we first copy it to the host
            if( !data_on_host_ && comm_->size()>1 ){
                int to_send = send_perm_.size();

                // collect values to send into the device buffer
                device_buffer_.at(0,to_send-1) = vector_[tag].at(send_perm_);

                // copy the buffer from the device into the send buffer on the host
                device_buffer_.at(0,to_send-1).dump(send_buffer_[tag].data());
            }

            // make the send
            for( int i=0; i<neighbours(); i++ ){
                int n = pattern().neighbour(i);

                if(data_on_host_){
                    send_requests_[tag][i] = comm_->Isend( send_buffer_[tag].data(),
                                                           n,
                                                           tag,
                                                           send_type_[n].MPIType );
                    recv_requests_[tag][i] = comm_->Irecv( recv_buffer_[tag].data(),
                                                           n,
                                                           tag,
                                                           recv_type_[n].MPIType );
                }else{
                    send_requests_[tag][i] = comm_->Isend( send_buffer_[tag].data()+send_offset_[n],
                                                           n,
                                                           tag,
                                                           send_type_[n].MPIType );
                    recv_requests_[tag][i] = comm_->Irecv( recv_buffer_[tag].data()+recv_offset_[n],
                                                           n,
                                                           tag,
                                                           recv_type_[n].MPIType );
                }
            }
        }

        busy_[tag] = true;
        
        return tag;
    }

    // complete pending communication associated with vector with tag
    template<typename Coord, typename Type>
    int Communicator<Coord, Type>::recv( int tag ){

        // check that we have been asked for a valid tag
        assert( std::find(vectors_.begin(), vectors_.end(), tag)!=vectors_.end() );

        // assert that we are not requesting a recv on a vector that has no pending recv 
        assert( busy_[tag] );

        // only communicate if actually need to
        if( pattern().num_neighbours() ){
            int send_size = send_requests_[tag].size();
            int recv_size = recv_requests_[tag].size();
            std::vector<MPI_Status> tmpStatus;
            tmpStatus.resize(std::max(send_size, recv_size));

            comm_->Waitall(send_requests_[tag], tmpStatus);
            comm_->Waitall(recv_requests_[tag], tmpStatus);


            // if the data is meant to be on the device we need to copy it
            // there from the host buffer
            if( !data_on_host_ ){
                int to_recv = recv_perm_.size();
                device_buffer_.at(0,to_recv-1) = recv_buffer_[tag];
                vector_[tag].at(recv_perm_) = device_buffer_.at(0,to_recv-1);
            }
        }

        busy_[tag] = false;
        return tag;
    }

    // complete all pending communication associated with vectors
    template<typename Coord, typename Type>
    int Communicator<Coord, Type>::recv_all(){
        for( std::set<int>::iterator i=vectors_.begin(); i!=vectors_.end(); i++ ){
            int tag = *i;
            if( busy_[tag] ){
                recv(tag);
            }
        }
        return 0;
    }

    // return a reference to the Pattern
    template<typename Coord, typename Type>
    const mesh::Pattern& Communicator<Coord, Type>::pattern() const {
        assert(pattern_);
        return *pattern_;
    }

}// namespace fvm

#endif
