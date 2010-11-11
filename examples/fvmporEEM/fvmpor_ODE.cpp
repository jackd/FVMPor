#include "fvmpor_ODE.h"

#ifdef PRECON_DSS
    #include "preconditioner_dss.h"
#endif
#ifdef PRECON_PARMS
    #include "preconditioner_parms.h"
#endif

#include <fvm/fvm.h>
#include <fvm/solver.h>
#include <fvm/integrators/ida_integrator.h>
#include <mpi/mpicomm.h>
#include <mpi/ompaffinity.h>
#include <fvm/impl/communicators/communicator.h>
#include <util/solution.h>
#include <util/timer.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <cstdio>

int main(int argc, char* argv[]) {

    const char* usage = " meshfile finalTime [outfile]\n";

    const double abstol = 5.0e-4;
    const double reltol = 5.0e-4;

    using namespace fvmpor;
try {
    // Initialise MPI
    mpi::Process process(argc, argv);
    mpi::MPICommPtr mpicomm( new mpi::MPIComm(MPI_COMM_WORLD, "WORLD") );

    // set omp affinity
    mpi::OMPAffinity omp_affinity;
    std::vector<int> my_cores( omp_affinity.get_cores(mpicomm) );
    int num_threads = omp_affinity.max_threads();
    std::vector<int> cores;
    assert(num_threads<=my_cores.size());
    for(int i=0; i<num_threads; i++)
        cores.push_back(my_cores[i]);
    omp_affinity.set_affinity(cores);

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

#ifdef PRECON
    typedef fvm::IDAIntegrator<Physics, Preconditioner> Integrator;
    typedef fvm::Solver<Physics, Integrator> Solver;
    Physics physics;
    *mpicomm << "initialised physics" << std::endl;
    Preconditioner preconditioner;
    *mpicomm << "initialised preconditioner" << std::endl;
    Integrator integrator(mesh, physics, preconditioner, reltol, abstol);
    *mpicomm << "initialised integrator" << std::endl;
    Solver solver(mesh, physics, integrator);
    *mpicomm << "initialised solver" << std::endl;
#else
    // DOM TIM - this is the code that you will use, because
    // we will have no need for a preconditioner
    typedef fvm::IDAIntegrator<Physics> Integrator;
    //typedef fvm::EEMIntegrator<Physics> Integrator; ?????

    typedef fvm::Solver<Physics, Integrator> Solver;
    Physics physics;
    *mpicomm << "initialised physics" << std::endl;
    Integrator integrator(mesh, physics, reltol, abstol);
    *mpicomm << "initialised integrator" << std::endl;
    Solver solver(mesh, physics, integrator);
    *mpicomm << "initialised solver" << std::endl;
#endif

    // DOM TIM - these options probably won't be available on the EEM
    // integrator (well, the order option certainly won't be)
    double maxTimestep = 0.;
    int maxOrder = 3;
    if(maxTimestep>0.)
        integrator.set_max_timestep(maxTimestep);
    if(maxOrder!=5)
        integrator.set_max_order(maxOrder);

    // find the output filename
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
        solution.add( t0, solver.begin(), solver.end_ext() );
        solution.write_timestep_VTK_XML( 0, mesh, filename );
    }
    // DOM TIM - here is where I choose the number of timesteps
    // I think it is a good idea to not change this interface, and provide
    // a method for interpolation between timesteps to find output
    // solutions at user-specified points
    int nt = 11;

    // timestep the solution
    *mpicomm << "beginning timestepping" << std::endl;
    double dt = (final_time-t0)/double(nt);
    double nextTime = t0 + dt;
    util::Timer timer;
    timer.tic();
    for( int i=0; i<nt; i++ )
    {
        if (mpicomm->rank() == 0)
            std::cout << "starting timestep at time " << nextTime-dt << "( " << solver.time() << ")" << std::endl;

        // advance the solution to nextTime
        solver.advance(nextTime);

        // save solution for output
        if(output_run){
            solution.add( nextTime, solver.begin(), solver.end_ext() );
            solution.write_timestep_VTK_XML( i+1, mesh, filename );
        }
        nextTime = t0 + (double)(i+2)*dt;
    }
    double finalTime = timer.toc();
    if( mpicomm->rank()==0)
        std::cout << std::endl << "Simulation took : " << finalTime << " seconds" << std::endl;

    // Output solver stats
    if (mpicomm->rank() == 0) {
        std::cout << "Physics calls = "
                  << physics.calls() << std::endl;
    }

} catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
}
}
