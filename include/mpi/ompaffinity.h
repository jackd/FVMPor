#ifndef OMPAFFINITY_H
#define OMPAFFINITY_H

#define WITH_MPI
#define MPI_DEBUG

// redirect to do nothing header if we aren't using MPI
#ifndef WITH_MPI
#include <mpi/mpicomm_null.h>
#else
#include <mpi/mpicomm.h>
#endif

#include <omp.h>
//#include <set.h>
//#include <vector.h>

namespace mpi {

// Identifies process id, number of processes, and communicator.
class OMPAffinity {
private:
public:
    OMPAffinity(){};

    int num_threads(){
        return omp_get_num_threads();
    };

    int max_threads(){
        int threads = omp_get_max_threads();
        return threads;
    };

    // returns a vector of the physical CPU cores that are available
    // for this MPI process
    std::vector< std::set<int> > get_affinity(MPICommPtr comm){
        int num_threads = omp_get_max_threads();
        std::vector< std::set<int> > thread_cores(num_threads);
#pragma omp parallel shared(num_threads, thread_cores)
        {
            int omp_rank = omp_get_thread_num();
            int omp_size = omp_get_num_threads();
            int omp_procs = omp_get_num_procs();

            // ensure that maximum number of threads available is being used
            // this is assumed for the time being, but could be relaxed
            // by adding some more logic below
            assert(omp_size==num_threads);

            kmp_affinity_mask_t mask_check;
            kmp_create_affinity_mask(&mask_check);
            assert(!kmp_get_affinity(&mask_check));
            std::set<int> my_cores;
            //for(int i=0; i<omp_get_num_procs()*m.mpicomm()->size(); i++)
            // check each OS thread for affinity
            // this is hard coded for the z5 which has 16 processes
            // on 8 cores, where core c has the thread set {c, c+8}
            for(int i=0; i<16; i++)
                if(kmp_get_affinity_mask_proc(i, &mask_check))
                    my_cores.insert(i%8);
            thread_cores[omp_rank] = my_cores;
        }

        return thread_cores;
    }

    
    std::vector<int> get_cores(MPICommPtr comm){
        std::vector<std::set<int> > thread_cores( get_affinity(comm) );

        // find the set of all cores that are available
        std::set<int> cores;
        for(int i=0; i<thread_cores.size(); i++){
            cores.insert(thread_cores[i].begin(), thread_cores[i].end());
        }

        // save the cores in a vector for returning
        std::vector<int> core_vector;
        std::set<int>::iterator it = cores.begin();
        std::set<int>::iterator end = cores.end();
        for( ; it!=end; it++ )
            core_vector.push_back(*it);

        // output the results to the diagnostic stream
        (*comm) << "===================================" << std::endl
                << "I have the following cores : ";
        for(it = cores.begin(); it!=end; it++)
            (*comm) << *it << " ";
        (*comm) << std::endl
                << "===================================" << std::endl;

        return core_vector;
    };

    // Set the affinity of OMP threads to specific cores.
    // Note that we are associating threads to cores, not OS processes.
    // On a hyper-threaded CPU there may be more than one OS process for each
    // core, however the algorithms we use are all CPU bound, so setting
    // more than one thread to have affinity with a core is counter-productive
    void set_affinity(std::vector<int> cores){
        int num_threads = omp_get_max_threads();
        // insist that a core is specified for each available thread
        assert(cores.size()==num_threads);
#pragma omp parallel shared(num_threads, cores)
        {
            int omp_rank = omp_get_thread_num();
            int omp_size = omp_get_num_threads();
            int omp_procs = omp_get_num_procs();
            int core = cores[omp_rank];

            // insist that the number of threads matches the maximum
            assert(omp_size==num_threads);
            
            kmp_affinity_mask_t mask;
            kmp_create_affinity_mask(&mask);
            // hard-coded for the z5
            kmp_set_affinity_mask_proc(core, &mask);
            kmp_set_affinity_mask_proc(core+8, &mask);
            assert(!kmp_set_affinity(&mask));
        }
    };
};


} // end namespace mpi


#endif // end header

