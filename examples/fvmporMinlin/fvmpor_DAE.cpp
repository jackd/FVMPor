#include "fvmpor_DAE.h"

#ifdef PRECON_DSS
    #include "preconditioner_dss_DAE.h"
#endif
#ifdef PRECON_PARMS
    #include "preconditioner_parms_DAE.h"
#endif

#include <fvm/fvm.h>
#include <fvm/solver.h>
#include <fvm/integrators/ida_integrator.h>
#include <mpi/mpicomm.h>
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

void print_ida_stats(void *ida_mem)
{
    long lenrw, leniw ;
    long lenrwLS, leniwLS;
    long nst, nfe, nsetups, nni, ncfn, netf;
    long nli, npe, nps, ncfl, nfeLS;
    int flag;

    flag = IDAGetWorkSpace(ida_mem, &lenrw, &leniw);
    assert(flag == 0);
    flag = IDAGetNumSteps(ida_mem, &nst);
    assert(flag == 0);
    flag = IDAGetNumResEvals(ida_mem, &nfe);
    assert(flag == 0);
    flag = IDAGetNumLinSolvSetups(ida_mem, &nsetups);
    assert(flag == 0);
    flag = IDAGetNumErrTestFails(ida_mem, &netf);
    assert(flag == 0);
    flag = IDAGetNumNonlinSolvIters(ida_mem, &nni);
    assert(flag == 0);
    flag = IDAGetNumNonlinSolvConvFails(ida_mem, &ncfn);
    assert(flag == 0);
    flag = IDASpilsGetWorkSpace(ida_mem, &lenrwLS, &leniwLS);
    assert(flag == 0);
    flag = IDASpilsGetNumLinIters(ida_mem, &nli);
    assert(flag == 0);
    flag = IDASpilsGetNumPrecEvals(ida_mem, &npe);
    assert(flag == 0);
    flag = IDASpilsGetNumPrecSolves(ida_mem, &nps);
    assert(flag == 0);
    flag = IDASpilsGetNumConvFails(ida_mem, &ncfl);
    assert(flag == 0);
    flag = IDASpilsGetNumResEvals(ida_mem, &nfeLS);
    assert(flag == 0);

    using std::printf;
    fprintf(stdout, "\nFinal Statistics.. \n\n");
    fprintf(stdout, "lenrw   = %5ld     leniw   = %5ld\n", lenrw, leniw);
    fprintf(stdout, "lenrwLS = %5ld     leniwLS = %5ld\n", lenrwLS, leniwLS);
    fprintf(stdout, "nst     = %5ld\n"                  , nst);
    fprintf(stdout, "nfe     = %5ld     nfeLS   = %5ld\n"  , nfe, nfeLS);
    fprintf(stdout, "nni     = %5ld     nli     = %5ld\n"  , nni, nli);
    fprintf(stdout, "nsetups = %5ld     netf    = %5ld\n"  , nsetups, netf);
    fprintf(stdout, "npe     = %5ld     nps     = %5ld\n"  , npe, nps);
    fprintf(stdout, "ncfn    = %5ld     ncfl    = %5ld\n\n", ncfn, ncfl);
}

int main(int argc, char* argv[]) {

    const char* usage = " meshfile finalTime [outfile]\n";

    const double abstol = 1.0e-3;
    const double reltol = 1.0e-3;

    using namespace fvmpor;
try {
    // Initialise MPI
    mpi::Process process(argc, argv);
    mpi::MPICommPtr mpicomm( new mpi::MPIComm(MPI_COMM_WORLD, "WORLD") );

    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << usage << std::endl;
        return EXIT_FAILURE;
    }

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
    Preconditioner preconditioner(physics);
    Integrator integrator(mesh, physics, preconditioner, reltol, abstol);
    Solver solver(mesh, physics, integrator);
#else
    typedef fvm::IDAIntegrator<Physics> Integrator;
    typedef fvm::Solver<Physics, Integrator> Solver;
    Physics physics;
    Integrator integrator(mesh, physics, reltol, abstol);
    Solver solver(mesh, physics, integrator);
#endif

    double maxTimestep = 0.;
    int maxOrder = 3;
    if(maxTimestep>0.)
        integrator.set_max_timestep(maxTimestep);
    if(maxOrder!=5)
        integrator.set_max_order(maxOrder);

    std::string filename;
    bool output_run = false;
    if(argc == 4){
        output_run = true;
        filename = std::string(argv[3]);
    }

    // save the initial conditions
    util::Solution<hM> solution(mpicomm);
    double t0 = solver.time();
    if(output_run){
        solution.add( t0, solver.begin(), solver.end_ext() );
        solution.write_timestep_VTK_XML( 0, mesh, filename );
    }
    int nt = 11;

    // initialise mass balance stats, and store for t0
    DoubleVector fluid_mass(nt+1);
    DoubleVector time_vec(nt+1);
    fluid_mass[0] = physics.compute_mass(mesh, solver.begin());
    time_vec[0] = t0;

    // timestep the solution
    double dt = (final_time-t0)/double(nt);
    double nextTime = t0 + dt;
    double startTime = MPI_Wtime();
    for( int i=0; i<nt; i++ )
    {
        if( mpicomm->rank() == 0 )
            std::cout << "starting timestep at time " << nextTime-dt << "( " << solver.time() << ")" << std::endl;

        // advance the solution to nextTime
        solver.advance(nextTime);

        // save solution for output
        if(output_run){
            solution.add( nextTime, solver.begin(), solver.end_ext() );
            solution.write_timestep_VTK_XML( i+1, mesh, filename );
        }
        fluid_mass[i+1] = physics.compute_mass(mesh, solver.begin());
        time_vec[i+1] = nextTime;
        nextTime = t0 + (double)(i+2)*dt;
    }
    double finalTime = MPI_Wtime() - startTime;
    if( mpicomm->rank()==0)
        std::cout << std::endl << "Simulation took : " << finalTime << " seconds" << std::endl;

    if( mpicomm->size()==1 ){
        // open file for output of stats
        std::ofstream mfid;
        std::string mfile_name;
        std::string runname("run");
        mfile_name = filename + ".m";
        mfid.open(mfile_name.c_str());
        assert( mfid );

        // output basic stats to file
        mfid << "maxOrder = " << maxOrder << ";" << std::endl;

        // perform mass-balance caluclations
        double flux_per_time = physics.mass_flux_per_time(mesh);
        mfid << "massError_" << runname  << " = [ ";
        for( int i=1; i<nt+1; i++ ){
            double time_elapsed = time_vec[i]-time_vec[0];
            double mass_balance = fluid_mass[i] - fluid_mass[0];
            double mass_error = fabs(mass_balance - time_elapsed*flux_per_time)/fluid_mass[i];
            mfid <<  mass_error << " ";
        }
        mfid << "];" << std::endl;
        // output step orders
        const std::vector<int>& orders = integrator.step_orders();
        mfid << "orders_" << runname  << " = [";
        for( int i=0; i<orders.size(); i++ )
            mfid << orders[i] << " ";
        mfid << "];" << std::endl;
        // output step sizes
        const std::vector<double>& sizes = integrator.step_sizes();
        mfid << "stepSizes_" << runname  << " = [";
        for( int i=0; i<sizes.size(); i++ )
            mfid << sizes[i] << " ";
        mfid << "];" << std::endl;
        mfid << "Feval_" << runname  << " = " << physics.calls() << ";" << std::endl;
        mfid.close();
    }

    // output the solution to file
    if(output_run && mpicomm->size()==1){
        solution.write_to_file( filename + ".run" );
    }

    // Output solver stats
    if (mpicomm->rank() == 0) {
        print_ida_stats(integrator.ida());
        std::cout << "Physics calls = "
                  << physics.calls() << std::endl;
#ifdef PRECON
        std::cout << "Preconditioner setups = "
                  << preconditioner.setups() << std::endl;
        std::cout << "Preconditioner callbacks = "
                  << preconditioner.callbacks() << std::endl;
        std::cout << "Preconditioner applications = "
                  << preconditioner.applications() << std::endl;
#endif
    }

} catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
}
}
