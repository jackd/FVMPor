#ifndef IDA_INTEGRATOR_H
#define IDA_INTEGRATOR_H

#include <fvm/fvm.h>
#include <fvm/mesh.h>
#include <mpi/mpicomm.h>

#include <idas/idas.h>
#include <idas/idas_spgmr.h>
#include <nvector/nvector_parallel.h>

#include <algorithm>
#include <cassert>
#include <vector>

namespace fvm {

// This class is uninstantiatable
template<class Physics>
class NoPreconditioner {
    NoPreconditioner();
    typedef typename Physics::TVec TVec;
    typedef typename Physics::TVecDevice TVecDevice;
public:
    typedef typename Physics::value_type value_type;
    typedef typename fvm::Callback<Physics> Callback;
    int setup(const mesh::Mesh&, double, double, double,
              const TVecDevice &,  const TVecDevice &,
              TVecDevice &, TVecDevice &, TVecDevice &, TVecDevice &, TVecDevice &, Callback)
    { return 0; }

    int apply(const mesh::Mesh&, double, double, double, double,
              const TVecDevice &, const TVecDevice &, const TVecDevice &,
              TVecDevice &, TVecDevice &, TVecDevice &, TVecDevice &, Callback)
    { return 0; }
};

template<class Physics, class Preconditioner = NoPreconditioner<Physics> >
class IDAIntegrator {
public:
    typedef mesh::Mesh Mesh;

    typedef typename Physics::value_type value_type;
    typedef typename fvm::Callback<Physics> Callback;
    typedef typename Physics::TVecDevice TVecDevice;
    typedef typename Physics::TVec TVec;

    IDAIntegrator(const Mesh& mesh, Physics& ph, double rtol, double atol);
    IDAIntegrator(const Mesh& mesh, Physics& ph, Preconditioner& pc, double rtol, double atol);
    ~IDAIntegrator();

    void initialise(double& t, TVecDevice &u, TVecDevice &up, Callback compute_residual);

    const Mesh& mesh() const;
    Preconditioner& preconditioner();

    void advance();                 // by one internal timestep
    void advance(double next_time); // to specified time

    // Returns the absolute tolerance
    value_type& abstol(int);
    value_type abstol(int) const;

    // Returns the relative tolerance
    double& reltol();
    double reltol() const;

    // Sets integration tolerances
    void set_tolerances();

    // set the maximum timestep taken by IDA
    void set_max_timestep(double);
    void set_max_order(int);
    double max_timestep() const;
    int max_order() const;
    void set_algebraic_variables(const TVec &vals);
    void compute_initial_conditions(TVecDevice &u0, TVecDevice &up0);

    // return pointer to the step orders
    const std::vector<int>& step_orders() const{
        return step_orders_;
    }

    // return pointer to the step orders
    const std::vector<double>& step_sizes() const{
        return step_sizes_;
    }

    // Exposes the IDA data structure
    void* ida();
    
private:
    IDAIntegrator(const IDAIntegrator&);
    IDAIntegrator& operator=(const IDAIntegrator&);

    const Mesh& m;
    Physics& physics;
    Preconditioner* pc;
    mpi::MPICommPtr procinfo;
    Callback compute_residual;
    double* t;
    void* ida_mem;
    double rtol;
    double atol;
    double max_timestep_;
    int max_order_;
    bool variableids_set_;
    N_Vector atolv;
    N_Vector weights;
    N_Vector ulocal;
    N_Vector uplocal;
    N_Vector uinterp;
    N_Vector upinterp;
    N_Vector variableids; // specify algebraic/differential variables
    TVecDevice u;
    TVecDevice up;
    TVecDevice ulocal_store;
    TVecDevice uplocal_store;
    TVecDevice weights_store;
    TVecDevice variableids_store;
    TVecDevice atolv_store;
    TVecDevice upinterp_store;
    TVecDevice uinterp_store;

    std::vector<int> step_orders_;
    std::vector<double> step_sizes_;
    static const int variables_per_node = VariableTraits<value_type>::number;

    // DEVICE
    //void copy_vector(N_Vector y, iterator w);
    void copy_vector(N_Vector y, TVecDevice &w);

    // IDA residual function
    static int f(double t,
                 N_Vector y, N_Vector yp, N_Vector r,
                 void*);

    // IDA preconditioner setup function
    static int psetup(double t,
                      N_Vector y, N_Vector yp, N_Vector r,
                      double c_j,
                      void*,
                      N_Vector t1, N_Vector t2, N_Vector t3);

    // IDA preconditioner solve function
    static int psolve(double t,
                      N_Vector y, N_Vector yp, N_Vector fy,
                      N_Vector r, N_Vector z,
                      double c_j, double delta,
                      void*,
                      N_Vector tmp);

};

template<class Physics, class Preconditioner>
IDAIntegrator<Physics, Preconditioner>::
IDAIntegrator(const Mesh& mesh, Physics& physics, double rtol, double atol)
    : m(mesh), physics(physics), pc(), t(), ida_mem(), rtol(rtol), atol(atol), max_timestep_(0.), max_order_(5), variableids_set_(false) {
    procinfo = m.mpicomm()->duplicate("IDA");
}

template<class Physics, class Preconditioner>
IDAIntegrator<Physics, Preconditioner>::
IDAIntegrator(const Mesh& mesh, Physics& physics, Preconditioner& pc, double rtol, double atol)
    : m(mesh), physics(physics), pc(&pc), t(), ida_mem(), rtol(rtol), atol(atol), max_timestep_(0.), max_order_(5), variableids_set_(false) {
    procinfo = m.mpicomm()->duplicate("IDA");
}

template<class Physics, class Preconditioner>
void IDAIntegrator<Physics, Preconditioner>::
initialise(double& tt, TVecDevice &y, TVecDevice &yp, Callback callback)
{
    *procinfo << "\tIDAIntegrator<Physics, Preconditioner>::initialise()" << std::endl;
    t = &tt;

    int localSize = mesh().local_nodes()*variables_per_node;
    int globalSize = mesh().global_nodes()*variables_per_node;

    u = TVecDevice(mesh().nodes()*variables_per_node, y.data());
    up = TVecDevice(mesh().nodes()*variables_per_node, yp.data());
    compute_residual = callback;

    // Initialise solution vectors
    // initialise with the passed values
    ulocal_store  = TVecDevice(localSize);
    uplocal_store = TVecDevice(localSize);
    ulocal_store.at(lin::all)  = u.at(0,localSize-1);
    uplocal_store.at(lin::all) = up.at(0,localSize-1);
    ulocal = N_VNewEmpty_Parallel( procinfo->communicator(),
                                     localSize, globalSize);
    assert(ulocal);
    uplocal = N_VNewEmpty_Parallel( procinfo->communicator(),
                                     localSize, globalSize);
    assert(uplocal);
    N_VSetArrayPointer_Parallel(ulocal_store.data(), ulocal);
    N_VSetArrayPointer_Parallel(uplocal_store.data(), uplocal);

    // Initialise interpolation vectors used when interpolation is performed
    // on output from IDA, so as not to overwrite previously calculated solutions
    uinterp = N_VNewEmpty_Parallel( procinfo->communicator(),
                                    localSize, globalSize );
    assert(uinterp);
    upinterp = N_VNewEmpty_Parallel( procinfo->communicator(),
                                     localSize, globalSize );
    assert(upinterp);
    // point the output of the interpolation to go directly
    // into u and up
    N_VSetArrayPointer_Parallel(u.data(), uinterp);
    N_VSetArrayPointer_Parallel(up.data(), upinterp);

    // Initialise weights vector
    weights = N_VNewEmpty_Parallel( procinfo->communicator(),
                                    localSize, globalSize );
    assert(weights);
    weights_store = TVecDevice(localSize);
    N_VSetArrayPointer_Parallel(weights_store.data(), weights);

    // Initialise absolute tolerances vector
    atolv = N_VNewEmpty_Parallel( procinfo->communicator(),
                                  localSize, globalSize );
    assert(atolv);
    atolv_store = TVecDevice(localSize, atol);
    N_VSetArrayPointer_Parallel(atolv_store.data(), atolv);

    // vector for tagging algebraic and differential variables
    variableids = N_VNewEmpty_Parallel( procinfo->communicator(),
                                        localSize, globalSize );
    assert(variableids);
    variableids_store = TVecDevice(localSize);
    N_VSetArrayPointer_Parallel(variableids_store.data(), variableids);

    // Create IDA data structure
    ida_mem = IDACreate();
    assert(ida_mem);

    // Initialise IDA internal memory
    int flag = IDAInit(
        ida_mem,
        reinterpret_cast<IDAResFn>(f),
        tt, ulocal, uplocal);
    assert(flag == IDA_SUCCESS);

    // Set default integration tolerances
    flag = IDASStolerances(ida_mem, rtol, atol);
    assert(flag == IDA_SUCCESS);

    // Set f_data parameter to be "this"
    flag = IDASetUserData(ida_mem, this);
    assert(flag == IDA_SUCCESS);

    // Initialise linear solver
    flag = IDASpgmr(ida_mem, 0);
    assert(flag == IDASPILS_SUCCESS);
    flag = IDASpilsSetGSType(ida_mem, MODIFIED_GS);
    assert(flag == IDASPILS_SUCCESS);

    // Initialise preconditioner
    if (pc) {
        flag = IDASpilsSetPreconditioner(
            ida_mem,
            reinterpret_cast<IDASpilsPrecSetupFn>(psetup),
            reinterpret_cast<IDASpilsPrecSolveFn>(psolve)
        );
        assert(flag == IDA_SUCCESS);
    }
}

template<class Physics, class Preconditioner>
IDAIntegrator<Physics, Preconditioner>::~IDAIntegrator()
{
    if (ida_mem) {
        N_VDestroy_Parallel(ulocal);
        N_VDestroy_Parallel(uplocal);
        N_VDestroy_Parallel(uinterp);
        N_VDestroy_Parallel(upinterp);
        N_VDestroy_Parallel(weights);
        N_VDestroy_Parallel(atolv);
        N_VDestroy_Parallel(variableids);
        IDAFree(&ida_mem);
    }
}

template<class Physics, class Preconditioner>
const mesh::Mesh& IDAIntegrator<Physics, Preconditioner>::mesh() const {
    return m;
}

template<class Physics, class Preconditioner>
Preconditioner& IDAIntegrator<Physics, Preconditioner>::preconditioner() {
    assert(pc);
    return *pc;
}

// Advances solution by one internal timestep
template<class Physics, class Preconditioner>
void IDAIntegrator<Physics, Preconditioner>::advance() {

    // we copy ulocal into u because the ulocal contains the current
    // solution value inside IDA, whereas u and up may contain a version
    // of the solution that was interpolated backwards
    // make this copy to ensure that up to date values are used for calculating
    // preprocess_timestep()
    u.at(0,mesh().local_nodes()-1) = ulocal_store;
    up.at(0,mesh().local_nodes()-1) = uplocal_store;

    physics.preprocess_timestep( *t, m, u, up );

    int flag = IDASolve( ida_mem, 1.0, t, ulocal, uplocal, IDA_ONE_STEP);
    assert(flag == IDA_SUCCESS);

    if( procinfo->rank()==0 )
        std::cerr << ".";

    // save the order and size of last step just completed
    int order_last;
    flag = IDAGetLastOrder(ida_mem, &order_last);
    step_orders_.push_back(order_last);
    double step_last;
    flag = IDAGetLastStep(ida_mem, &step_last);
    step_sizes_.push_back(step_last);
}

// Advances solution to the specified time
template<class Physics, class Preconditioner>
void IDAIntegrator<Physics, Preconditioner>::advance(double next_time) {

    // advance the solution to next_time
    while( (*t)<next_time )
        advance();

    // Get IDA to interpolate the solution backwards from t to next_time
    // this doesn't change the internal state of IDA, it is simply
    // to ensure that the solution returned to the user is that at the
    // requested time.
    // The vector uinterp and upinterp are pointed directly into u
    int flag = IDAGetDky( ida_mem, next_time, 0, uinterp );
    assert(flag == IDA_SUCCESS);
    flag = IDAGetDky( ida_mem, next_time, 1, upinterp );
    assert(flag == IDA_SUCCESS);
}

// Returns the absolute tolerance
template<class Physics, class Preconditioner>
typename IDAIntegrator<Physics, Preconditioner>::value_type&
IDAIntegrator<Physics, Preconditioner>::abstol(int i)
{
    assert(ida_mem);
    assert(i >= 0 && i < mesh().local_nodes());
    return atolv_store[i];
}

template<class Physics, class Preconditioner>
typename IDAIntegrator<Physics, Preconditioner>::value_type
IDAIntegrator<Physics, Preconditioner>::abstol(int i) const
{
    assert(ida_mem);
    return atolv_store[i];
}

// Returns the relative tolerance
template<class Physics, class Preconditioner>
double& IDAIntegrator<Physics, Preconditioner>::reltol() {
    return rtol;
}

template<class Physics, class Preconditioner>
double IDAIntegrator<Physics, Preconditioner>::reltol() const {
    return rtol;
}

// Sets integration tolerances
template<class Physics, class Preconditioner>
void IDAIntegrator<Physics, Preconditioner>::set_tolerances() {
    int flag = IDASVtolerances(ida_mem, rtol, atolv);
    assert(flag == IDA_SUCCESS);
}

// set the variable ids
// this allows the user to specify which variables are algabraic and which
// are differential.
// vals[i]=0. -> variable i is algebraic
// vals[i]=1. -> variable i is differntial
template<class Physics, class Preconditioner>
// DEVICE
//void IDAIntegrator<Physics, Preconditioner>::set_algebraic_variables(const std::vector<double> &vals){
void IDAIntegrator<Physics, Preconditioner>::set_algebraic_variables(const TVec &vals){
    // sanity check the input
    assert(vals.size()==variables_per_node*m.local_nodes());
    for(int i=0; i<vals.size(); i++)
        assert(vals[i]==0. || vals[i]==1.);

    // DEVICE
    // copy user specified variable ids into NV_Vector
    //double* dest = reinterpret_cast<double*>(NV_DATA_P(variableids));
    //std::copy(vals.begin(), vals.end(), dest);
    variableids_store = vals;
    // call IDA to set the ids
    int flag = IDASetId(ida_mem, variableids);
    assert(flag == IDA_SUCCESS);
    variableids_set_=true;
}

// compute consistent initial conditions
// if IDAIntegrator::set_algebraic_variables() has been called the method
// will attempt to find derivatives for the differential variabels
// and values for the algebraic variables
template<class Physics, class Preconditioner>
void IDAIntegrator<Physics, Preconditioner>::compute_initial_conditions(TVecDevice &u0, TVecDevice &up0){
    int icopt = IDA_Y_INIT;
    if(variableids_set_){
        icopt = IDA_YA_YDP_INIT;
    }

    int flag = IDACalcIC(ida_mem, icopt, (*t)+1.);
    assert(flag==IDA_SUCCESS);

    // DEVICE
    // get the initial conditions
    N_Vector yy0_mod, yp0_mod;
    //yy0_mod = N_VNew_Parallel(
    yy0_mod = N_VNewEmpty_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(yy0_mod);
    N_VSetArrayPointer_Parallel(u0.data(), yy0_mod);

    //yp0_mod = N_VNew_Parallel(
    yp0_mod = N_VNewEmpty_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(yp0_mod);
    N_VSetArrayPointer_Parallel(up0.data(), yp0_mod);

    flag = IDAGetConsistentIC(ida_mem, yy0_mod, yp0_mod);
    // DEVICE
    // no need for copy
    /*
    std::copy(
        reinterpret_cast<value_type*>(NV_DATA_P(yy0_mod)),
        reinterpret_cast<value_type*>(NV_DATA_P(yy0_mod)) + variables_per_node*mesh().local_nodes(),
        u0
    );
    std::copy(
        reinterpret_cast<value_type*>(NV_DATA_P(yp0_mod)),
        reinterpret_cast<value_type*>(NV_DATA_P(yp0_mod)) + variables_per_node*mesh().local_nodes(),
        up0
    );
    */
}

// set the maximum timestep taken by IDA
template<class Physics, class Preconditioner>
void IDAIntegrator<Physics, Preconditioner>::set_max_timestep(double max_ts) {
    assert(max_ts>0.);
    max_timestep_ = max_ts;
    int flag = IDASetMaxStep(ida_mem, max_ts);
    assert(flag == IDA_SUCCESS);
}

// set the maximum order of BDF used by IDA
template<class Physics, class Preconditioner>
void IDAIntegrator<Physics, Preconditioner>::set_max_order(int max_order) {
    assert(max_order>0 && max_order<=5);
    max_order_ = max_order;
    int flag = IDASetMaxOrd(ida_mem, max_order);
    assert(flag == IDA_SUCCESS);
}

template<class Physics, class Preconditioner>
void* IDAIntegrator<Physics, Preconditioner>::ida() {
    return ida_mem;
}

// DEVICE
// it is a bit hard to see how this would work
// (IDA)
// Computes the residual function
template<class Physics, class Preconditioner>
int IDAIntegrator<Physics, Preconditioner>::
f(double t, N_Vector y, N_Vector yp, N_Vector res, void* ip)
{
    IDAIntegrator* integrator = static_cast<IDAIntegrator*>(ip);

    *integrator->t = t;

    int N = NV_LOCLENGTH_P(y);
    TVecDevice tmpu = TVecDevice(N, NV_DATA_P(y));
    TVecDevice tmpup = TVecDevice(N, NV_DATA_P(yp));
    integrator->u.at(0,N-1) = tmpu;
    integrator->up.at(0,N-1) = tmpup;

    TVecDevice r(N, NV_DATA_P(res));

    bool communicate = true;
    int success = integrator->compute_residual(r, communicate);

    return success;
}

// (IDA)
// Performs any processing that might be required to set up the preconditioner.
// An example would be forming an approximation to the Jacobian matrix
// J = dF/dy + c_j*dF/dyp and performing an LU factorisation on it.
// This function is called only as often as the underlying solver deems it
// necessary to achieve convergence.
template<class Physics, class Preconditioner>
int IDAIntegrator<Physics, Preconditioner>::
psetup(double tt, N_Vector y, N_Vector yp, N_Vector r,
       double c, void* ip,
       N_Vector t1, N_Vector t2, N_Vector t3)
{
    IDAIntegrator* integrator = static_cast<IDAIntegrator*>(ip);

    const mesh::Mesh& m = integrator->mesh();

    assert(NV_GLOBLENGTH_P(t1) == m.global_nodes() * variables_per_node);
    assert(NV_LOCLENGTH_P(t1) == m.local_nodes() * variables_per_node);
    assert(NV_GLOBLENGTH_P(t2) == m.global_nodes() * variables_per_node);
    assert(NV_LOCLENGTH_P(t2) == m.local_nodes() * variables_per_node);
    assert(NV_GLOBLENGTH_P(t3) == m.global_nodes() * variables_per_node);
    assert(NV_LOCLENGTH_P(t3) == m.local_nodes() * variables_per_node);

    double h;
    int flag = IDAGetCurrentStep(integrator->ida(), &h);
    assert(flag == 0);
    flag = IDAGetErrWeights(integrator->ida(), integrator->weights);
    assert(flag == 0);

    int N = NV_LOCLENGTH_P(r);
    TVecDevice res(N, NV_DATA_P(r));
    TVecDevice w(N, NV_DATA_P(integrator->weights));
    TVecDevice temp1(N, NV_DATA_P(t1));
    TVecDevice temp2(N, NV_DATA_P(t2));
    TVecDevice temp3(N, NV_DATA_P(t3));

    int result = integrator->preconditioner().setup(
        m, tt, c, h,
        res,
        w,
        integrator->u,
        integrator->up,
        temp1,
        temp2,
        temp3,
        integrator->compute_residual
    );

    return result;
}

// (IDA)
// Solves the linear system Pz = r, where P is the preconditioner matrix.
template<class Physics, class Preconditioner>
int IDAIntegrator<Physics, Preconditioner>::
psolve(double tt, N_Vector y, N_Vector yp, N_Vector r,
       N_Vector rr, N_Vector zz,
       double c, double delta,
       void* ip, N_Vector tmp)
{
    IDAIntegrator* integrator = static_cast<IDAIntegrator*>(ip);

    const mesh::Mesh& m = integrator->mesh();

    assert(NV_GLOBLENGTH_P(tmp) == m.global_nodes() * variables_per_node);
    assert(NV_LOCLENGTH_P(tmp) == m.local_nodes() * variables_per_node);

    double h;
    int flag = IDAGetCurrentStep(integrator->ida(), &h);
    assert(flag == 0);
    flag = IDAGetErrWeights(integrator->ida(), integrator->weights);
    assert(flag == 0);

    int N = NV_LOCLENGTH_P(r);
    TVecDevice res(N, NV_DATA_P(r));
    TVecDevice w(N, NV_DATA_P(integrator->weights));
    TVecDevice rhs(N, NV_DATA_P(rr));
    TVecDevice z(N, NV_DATA_P(zz));
    TVecDevice temp(N, NV_DATA_P(tmp));

    z.at(lin::all) = rhs;

    return integrator->preconditioner().apply(
        m, tt, c, h, delta,
        res, w, rhs, integrator->u, integrator->up, z, temp,
        integrator->compute_residual
    );
}

// Definition of static member
template<class Physics, class Preconditioner>
const int IDAIntegrator<Physics, Preconditioner>::variables_per_node;

} // end namespace fvm

#endif
