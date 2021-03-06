#include "fvmpor_ODE.h"

#include <fvm/fvm.h>
#include <fvm/solver.h>
//#include <fvm/integrators/ida_integrator.h>
#include <fvm/integrators/eem_integrator.h>
#include <mpi/mpicomm.h>
#include <mpi/ompaffinity.h>
#include <fvm/impl/communicators/communicator.h>
#include <util/solution.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <cstdio>

template<typename T> std::string to_string(const T& t){
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

int main(int argc, char* argv[]) {

    const char* usage = " meshfile finalTime [outfile]\n";

    const double abstol = 5.0e-4;
    const double reltol = 5.0e-4;

    using namespace fvmpor;
try {
    // Initialise MPI
    mpi::Process process(argc, argv);
    mpi::MPICommPtr mpicomm( new mpi::MPIComm(MPI_COMM_WORLD, "WORLD") );

    if(mpicomm->rank()==0){
        std::cerr << "==============================================================" << std::endl;
        std::cerr << "                      HARDWARE" << std::endl;
    }

    // set omp affinity
    mpi::OMPAffinity omp_affinity;
    std::vector<int> my_cores( omp_affinity.get_cores(mpicomm) );

    int num_threads = omp_affinity.max_threads();
    std::vector<int> cores;
    // ensure that the number of threads does not exceed the number of available cores
    if(num_threads>my_cores.size()){
        std::cerr << "FVMPor error : number of OpenMP threads must not exceed the number of available cores."
                  << std::endl
                  << "The number of threads must be set before calling FVMPor. For example :"
                  << std::endl
                  << "export OMP_NUM_THREADS=1"
                  << std::endl;
        exit(1);
    }
    for(int i=0; i<num_threads; i++)
        cores.push_back(my_cores[i]);
    omp_affinity.set_affinity(cores);

    if(mpicomm->rank()==0)
        std::cerr << "There are " << mpicomm->size() << " MPI processes, each with " << num_threads << " OpenMP threads" << std::endl;
    // initialise CUDA
#ifdef USE_CUDA
    {
            if(mpicomm->rank()==0)
                std::cout << "Initialising GPU... " << std::endl;
            int num_devices = lin::gpu::num_devices();
            int num_processes = mpicomm->size();
            int this_process = mpicomm->rank();
            assert(num_processes<=num_devices);
            lin::gpu::set_device(this_process);
            std::string device_name = lin::gpu::get_device_name();
            assert( cublasInit() == CUBLAS_STATUS_SUCCESS );
            for(int i=0; i<mpicomm->size(); i++){
                if(mpicomm->rank()==i)
                    std::cerr << "\tMPI process " << i << " using device " << this_process
                          << " : " << device_name << std::endl;
                mpicomm->barrier();
            }
    }
#endif

    if(mpicomm->rank()==0)
        std::cerr << "==============================================================" << std::endl;

    // verify that the user has passed enough command line arguments
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << usage << std::endl;
        return EXIT_FAILURE;
    }

    // determine the simulation time from command line
    std::istringstream timeString(argv[2]);
    double final_time;
    if( !(timeString >> final_time) || final_time<=0. ){
        std::cerr << "invalid final time as argument " << argv[2] << std::endl;
        return EXIT_FAILURE;
    }

    // Load mesh
    mesh::Mesh mesh(argv[1], mpicomm);
    *mpicomm << "loaded mesh" << std::endl;

    // EEMCHANGE
    // replace the IDA integrator with EEM version
    //typedef fvm::IDAIntegrator<Physics> Integrator;
    typedef fvm::EEMIntegrator<Physics> Integrator;
    typedef fvm::Solver<Physics, Integrator> Solver;
    Physics physics;
    *mpicomm << "initialised physics" << std::endl;
    Integrator integrator(mesh, physics, reltol, abstol);
    *mpicomm << "initialised integrator" << std::endl;
    Solver solver(mesh, physics, integrator);
    *mpicomm << "initialised solver" << std::endl;

    typedef typename Physics::TVec TVec;

    double maxTimestep = 0.;
    if(maxTimestep>0.)
        integrator.set_max_timestep(maxTimestep);

    *mpicomm << "set integrator max timestep (" << maxTimestep << ")" << std::endl;

    std::string filename;
    bool output_run = false;
    if(argc == 4){
        output_run = true;
        filename = std::string(argv[3]);
    }

    // save the initial conditions
    util::Solution<Head> solution(mpicomm);
    double t0 = solver.time();
    if(output_run){
        // make a temporary vector on the host because the solution
        // returned by the solver may be on the device
        TVec sol(solver.solution());
        solution.add( t0, sol );
        solution.write_timestep_VTK_XML( 0, mesh, filename );
    }
    //int nt = round(final_time/3600);
    int nt = 11;

    // initialise mass balance stats, and store for t0
    //DoubleVector fluid_mass(nt+1);
    DoubleVector time_vec(nt+1);
    //fluid_mass[0] = physics.compute_mass(mesh, solver.begin());
    time_vec[0] = t0;

    // timestep the solution
    *mpicomm << "beginning timestepping" << std::endl;
    double dt = (final_time-t0)/double(nt);
    double nextTime = t0 + dt;
    double startTime = MPI_Wtime();
    for( int i=0; i<nt; i++ )
    {
        if (mpicomm->rank() == 0)
            std::cout << "\nstarting timestep at time " << nextTime-dt << "( " << solver.time() << ")" << std::endl;

        // advance the solution to nextTime
        solver.advance(nextTime);

        // save solution for output
        if(output_run){
            TVec sol(solver.solution());
            solution.add( nextTime, sol );
            solution.write_timestep_VTK_XML( i+1, mesh, filename );
        }
        //fluid_mass[i+1] = physics.compute_mass(mesh, solver.begin());
        time_vec[i+1] = nextTime;
        nextTime = t0 + (double)(i+2)*dt;
    }
    double finalTime = MPI_Wtime() - startTime;
    if( mpicomm->rank()==0){
        std::cout << std::endl << "Simulation took : " << finalTime << " seconds" << std::endl;
    }

    // Output solver stats
    if (mpicomm->rank() == 0) {
        std::cout << "Physics calls = "
                  << physics.calls() << std::endl;
    }

} catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
}
}
