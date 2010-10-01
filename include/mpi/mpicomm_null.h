#ifndef MPICOMM_NULL_H
#define MPICOMM_NULL_H

#include <cassert>
#include <iomanip>

namespace mpi {

// Identifies process id, number of processes, and communicator.
class MPIComm {
public:
    MPIComm(int comm = 0) {
        rank_ = 1;
        size_ = 1;
        comm_ = comm;
    }

    int rank() const {
        return rank_;
    }

    int size() const {
        return size_;
    }

    int communicator() const
    {
        return comm_;
    }

    // duplicate the MPI communicator
    MPIComm duplicate() const {
        return(MPIComm(1));
    }

private:
    int rank_;
    int size_;
    int comm_;
};

// Singleton class that handles MPI startup and finalisation.
class Process {
public:
    Process(int& argc, char**& argv) {
        assert(!instantiated());
    }

    int rank() const {
        return mpicomm().rank();
    }

    int size() const {
        return mpicomm().size();
    }

    MPIComm mpicomm() const {
        static MPIComm mpicomm_;
        return mpicomm_;
    }
private:
    Process(const Process&);
    Process& operator=(const Process&);

    bool instantiated() {
        static int count_ = 0;
        return count_++;
    }
};

} // end namespace mpi
#endif
