#ifndef MPICOMM_H
#define MPICOMM_H

#define NEWCOMM
#define WITH_MPI
#define MPI_DEBUG

// redirect to do nothing header if we aren't using MPI
#ifndef WITH_MPI
#include <mpi/mpicomm_null.h>
#else

#include <mpi.h>
#include <boost/smart_ptr.hpp>
#include <util/streamstring.h>

#include <string>
#include <vector>
#include <exception>

namespace mpi {

// returns string describing MPI error specified by flag
std::string flag_string(int flag);

// traits class for determining MPI_Type flag
template  <typename T>
struct MPITraits{
    enum {MPIType = T::MPIType};
};
template  <>
struct MPITraits<int>{
    enum {MPIType = MPI_INT};
};
template  <>
struct MPITraits<double>{
    enum {MPIType = MPI_DOUBLE};
};
template  <>
struct MPITraits<float>{
    enum {MPIType = MPI_FLOAT};
};
template  <>
struct MPITraits<char>{
    enum {MPIType = MPI_CHAR};
};

// exception handler for MPIComm
class MPIException: public std::exception
{
    std::string error_message_;
    virtual const char* what() const throw()
    {
        if( error_message_.length()==0 )
            return "MPI exception";
        return error_message_.c_str();
    }

public:
    virtual ~MPIException()throw(){};
    void set_error_message( const std::string &str ){
        error_message_ = str;
    }
};

// Identifies process id, number of processes, and communicator.
class MPIComm {

    typedef boost::shared_ptr<MPIComm> MPICommPtr;
#ifdef MPI_DEBUG
    typedef std::ofstream logstream;
#else
    typedef util::onullstream logstream;
#endif
public:
    MPIComm(MPI_Comm comm, std::string name) {
        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &size_);
        MPI_Errhandler_set(comm, MPI_ERRORS_RETURN);
        comm_ = comm;
        name_ = name;

#ifdef MPI_DEBUG
        // open logfile
        std::string filename = "./output/MPI_"+name_+"_"+util::to_string(size_)+"_"+util::to_string(rank_)+".log";
        log_file_.open(filename.c_str());
        if( !log_file_.is_open() )
            throw_with_message( this->name() + " : unable to open log_file : " + filename  );
        log_file_ <<  "==================================" << std::endl << "creating comm " << this->name() << " = " << comm_ << std::endl << "==================================" << std::endl;
#endif
    }

    template<typename T>
    MPIComm& operator<<(const T& t) {
#ifdef MPI_DEBUG
        log_file_ << t;
#endif
        return *this;
    }
    MPIComm& operator<<(std::ostream& (*pfn)(std::ostream&)) {
#ifdef MPI_DEBUG
       log_file_ << pfn;
#endif
       return *this;
    }

    logstream& log_stream()
    {
        return log_file_;
    }

    ~MPIComm(){
        int result;
        MPI_Comm_compare( comm_, MPI_COMM_WORLD, &result);
        *this << "freeing comm " << name_ << "(" << comm_ << ")" << " that " << (result==MPI_IDENT ? "is ": "isn't ") << "MPI_COMM_WORLD" << std::endl;
        if(result!=MPI_IDENT){
#ifdef MPI_DEBUG
            if( log_file_.is_open() )
                log_file_.close();
#endif
            MPI_Comm_free(&comm_);
        }
    }

    int rank(){
        return rank_;
    }

    int size(){
        return size_;
    }

    MPI_Comm communicator()
    {
        return comm_;
    }

    const std::string name(){
        std::string my_name = name_+"("+util::to_string(rank_)+", "+util::to_string(size_)+")";
        return my_name;
    }

    const std::string name_short(){
        std::string my_name = name_;
        return my_name;
    }

    // send a single item of type T
    template<typename T>
    MPI_Request Isend( T dat, int destination, int tag ){
        *this << "Isend : destination " << destination << " tag " << tag << std::flush;
        MPI_Request request;
        MPI_Datatype data_type = MPITraits<T>::MPIType;
        int flag = MPI_Isend( reinterpret_cast<void *>(&dat), 1, data_type, destination, tag, comm_, &request );
        *this << "\t: sent with request " << request << " and flag " << flag_string(flag) << std::endl;
        if(flag!=MPI_SUCCESS)
            throw_with_message( this->name() + " : Isend() : flag = " + flag_string(flag), flag );
        return request;
    }

    // receive a single item of type T
    template<typename T>
    MPI_Request Irecv( T &dat, int source, int tag ){
        *this << "Irecv : source " << source << " tag " << tag << std::flush;
        MPI_Request request;
        MPI_Datatype data_type = MPITraits<T>::MPIType;
        int flag = MPI_Irecv( reinterpret_cast<void *>(&dat), 1, data_type, source, tag, comm_, &request );
        *this << "\t: received with request " << request << " and flag " << flag_string(flag) << std::endl;
        if(flag!=MPI_SUCCESS)
            throw_with_message( this->name() + " : Irecv() : flag = " + flag_string(flag), flag );
        return request;
    }

    // send a single item of type T
    template<typename T>
    MPI_Request Isend(T *dat, int destination, int tag, MPI_Datatype data_type ){
        *this << "Isend : destination " << destination << " tag " << tag << std::flush;
        MPI_Request request;
        int flag = MPI_Isend( reinterpret_cast<void *>(dat), 1, data_type, destination, tag, comm_, &request );
        *this << "\t: sent with request " << request << " and flag " << flag_string(flag) << std::endl;
        if(flag!=MPI_SUCCESS)
            throw_with_message( this->name() + " : Isend() : flag = " + flag_string(flag), flag );
        return request;
    }

    // receive a single item of type T
    template<typename T>
    MPI_Request Irecv( T *dat, int source, int tag, MPI_Datatype data_type ){
        *this << "Irecv : source " << source << " tag " << tag << std::flush;
        MPI_Request request;
        int flag = MPI_Irecv( reinterpret_cast<void *>(dat), 1, data_type, source, tag, comm_, &request );
        *this << "\t: received with request " << request << " and flag " << flag_string(flag) << std::endl;
        if(flag!=MPI_SUCCESS)
            throw_with_message( this->name() + " : Irecv() : flag = " + flag_string(flag), flag );
        return request;
    }

    // send a vector of type T
    template <typename T>
    MPI_Request Isend( std::vector<T>& buffer, int destination, int tag ){
        *this << "Isend : vector of length " << buffer.size() << ", desination " << destination << ", tag " << tag << std::flush;
        MPI_Request request;
        MPI_Datatype data_type = MPITraits<T>::MPIType;
        int flag = MPI_Isend( reinterpret_cast<void*>(&buffer[0]), buffer.size(), data_type, destination, tag, comm_, &request );
        *this << "\t: sent with request " << request  << " and flag " << flag_string(flag) << std::endl;
        if(flag!=MPI_SUCCESS)
            throw_with_message( this->name() + " : Isecv() : flag = " + flag_string(flag), flag );
        return request;
    }

    // recieve a vector of type T
    template <typename T>
    MPI_Request Irecv( std::vector<T>& buffer, int source, int tag ){
        *this << "Irecv : vector of length " << buffer.size() << ", source " << source << ", tag " << tag << std::flush;
        MPI_Request request;
        MPI_Datatype data_type = MPITraits<T>::MPIType;
        int flag = MPI_Irecv( reinterpret_cast<void*>(&buffer[0]), buffer.size(), data_type, source, tag, comm_, &request );
        *this << "\t: received with request " << request << " and flag " << flag_string(flag) << std::endl;
        if(flag!=MPI_SUCCESS)
            throw_with_message( this->name() + " : Irecv() : flag = " + flag_string(flag), flag );
        return request;
    }

    // wait for a send/receive to complete
    int Wait( MPI_Request* request, MPI_Status* status ){
        *this << "Wait : request " << *request << std::flush;
        int flag = MPI_Wait( request, status );
        *this << "\t: finished with flag " << flag_string(flag) << std::endl;
        if(flag!=MPI_SUCCESS){
            if( flag==MPI_ERR_IN_STATUS )
                flag = status->MPI_ERROR;
            throw_with_message( this->name() + " : Wait() : flag = " + flag_string(flag), flag );
        }
        return flag;
    }

    // wait for a send/receive to complete
    int Waitall( std::vector<MPI_Request> request, std::vector<MPI_Status> status ){
        *this << "Waitall : " << request.size() << " requests" << std::flush;
        assert(request.size() == status.size());
        int flag = MPI_Waitall( request.size(), const_cast<MPI_Request*>(&request[0]), const_cast<MPI_Status*>(&status[0]) );
        *this << "\t: finished with flag " << flag_string(flag) << std::endl;
        if(flag!=MPI_SUCCESS){
            if( flag==MPI_ERR_IN_STATUS ){
                int i=0;
                while( status[i++].MPI_ERROR==MPI_SUCCESS );
                flag = status[i-1].MPI_ERROR;
            }
            throw_with_message( this->name() + " : Waitall() : flag = " + flag_string(flag), flag );
        }
        return flag;
    }

    // barrier
    void Barrier(){
        *this << "MPI_Barrier" << std::endl;
        MPI_Barrier( comm_ );
        *this << "\tMPI_Barrier finished" << std::endl;
    }

    // duplicate the MPI communicator
    MPICommPtr duplicate( std::string name ) {
        // duplicate the MPI_comm using MPI calls
        MPI_Comm new_comm;
        MPI_Comm_dup(comm_, &new_comm);

        *this << "duplicating comm " << name_ << "(" << comm_ << ")" << " -> " << name << "(" << new_comm << ")" << std::endl;
        // store our MPIComm assocated with the new MPI_Comm
        MPICommPtr newComm = MPICommPtr( new MPIComm(new_comm, name) );
        return( newComm );
    }

private:
    int rank_;
    int size_;
    MPI_Comm comm_;
    std::string name_;
    logstream log_file_;
    MPIException MPIExcept;

    void throw_with_message( const std::string msg ){
        *this << "******************ERROR******************" << std::endl << msg << std::endl << "*****************************************" << std::endl;
        MPIExcept.set_error_message(msg);
        throw MPIExcept;
    }

    void throw_with_message( const std::string msg, int flag ){
        if( flag>=0 ){
            char error_string[MPI_MAX_ERROR_STRING];
            int length_of_error_string;

            MPI_Error_string(flag, error_string, &length_of_error_string);
            throw_with_message( msg+"\nMPI ERROR STRING : " + error_string);
        }else
            throw_with_message( msg );
    }
};

typedef boost::shared_ptr<MPIComm> MPICommPtr;

// Singleton class that handles MPI startup and finalisation.
class Process {
public:
    Process(int& argc, char**& argv) {
        assert(!instantiated());
        MPI_Init(&argc, &argv);
    }

    ~Process() {
        MPI_Finalize();
    }

private:
    Process(const Process&);
    Process& operator=(const Process&);

    bool instantiated() {
        static int count_ = 0;
        return count_++;
    }
};


// return a string that describes an MPI flag
inline
std::string flag_string(int flag){
    if( flag==MPI_SUCCESS )
        return std::string("MPI_SUCCESS");
    if(flag==MPI_ERR_REQUEST)
        return std::string("MPI_ERR_REQUEST");
    if(flag==MPI_ERR_ARG)
        return std::string("MPI_ERR_ARG");
    if(flag==MPI_ERR_IN_STATUS)
        return std::string("MPI_ERR_IN_STATUS");
    if(flag==MPI_ERR_BUFFER)
        return std::string("MPI_ERR_BUFFER");
    if(flag==MPI_ERR_COUNT)
        return std::string("MPI_ERR_COUNT");
    if(flag==MPI_ERR_TYPE)
        return std::string("MPI_ERR_TYPE");
    if(flag==MPI_ERR_TAG)
        return std::string("MPI_ERR_TAG");
    if(flag==MPI_ERR_COMM)
        return std::string("MPI_ERR_COMM");
    if(flag==MPI_ERR_RANK)
        return std::string("MPI_ERR_RANK");
    if(flag==MPI_ERR_ROOT)
        return std::string("MPI_ERR_ROOT");
    if(flag==MPI_ERR_GROUP)
        return std::string("MPI_ERR_GROUP");
    if(flag==MPI_ERR_OP)
        return std::string("MPI_ERR_OP");
    if(flag==MPI_ERR_TOPOLOGY)
        return std::string("MPI_ERR_TOPOLOGY");
    if(flag==MPI_ERR_DIMS)
        return std::string("MPI_ERR_DIMS");
    if(flag==MPI_ERR_UNKNOWN)
        return std::string("MPI_ERR_UNKNOWN");
    if(flag==MPI_ERR_TRUNCATE)
        return std::string("MPI_ERR_TRUNCATE");
    if(flag==MPI_ERR_OTHER)
        return std::string("MPI_ERR_OTHER");
    if(flag==MPI_ERR_INTERN)
        return std::string("MPI_ERR_INTERN");
    if(flag==MPI_ERR_PENDING)
        return std::string("MPI_ERR_PENDING");
    if(flag==MPI_ERR_LASTCODE)
        return std::string("MPI_ERR_LASTCODE");
    if(flag>MPI_ERR_LASTCODE)
        return std::string("INVALID ERROR CODE");

    return std::string("UNKNOWN_FLAG");
}


} // end namespace mpi

#endif // end USE_MPI

#endif // end header

