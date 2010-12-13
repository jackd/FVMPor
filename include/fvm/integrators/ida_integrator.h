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
public:
    typedef typename Physics::value_type value_type;
    typedef typename fvm::Callback<Physics> Callback;
    typedef typename fvm::Iterator<value_type>::type iterator;
    typedef typename fvm::ConstIterator<value_type>::type const_iterator;
    int setup(const mesh::Mesh&, double, double, double,
              const_iterator, const_iterator,
              iterator, iterator, iterator, iterator, iterator, Callback)
    { return 0; }

    int apply(const mesh::Mesh&, double, double, double, double,
              const_iterator, const_iterator, const_iterator,
              iterator, iterator, iterator, iterator, Callback)
    { return 0; }
};

template<class Physics, class Preconditioner = NoPreconditioner<Physics> >
class IDAIntegrator {
public:
    typedef mesh::Mesh Mesh;

    typedef typename Physics::value_type value_type;
    typedef typename fvm::Callback<Physics> Callback;
    typedef typename fvm::Iterator<value_type>::type iterator;
    typedef typename fvm::ConstIterator<value_type>::type const_iterator;

    //IDAIntegrator(const Mesh& mesh, double rtol, double atol);
    //IDAIntegrator(const Mesh& mesh, Preconditioner& pc, double rtol, double atol);
    IDAIntegrator(const Mesh& mesh, Physics& ph, double rtol, double atol);
    IDAIntegrator(const Mesh& mesh, Physics& ph, Preconditioner& pc, double rtol, double atol);
    ~IDAIntegrator();

    void initialise(double& t, iterator u, iterator up, Callback compute_residual);

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
    void set_algebraic_variables(const std::vector<double> &vals);
    void compute_initial_conditions(iterator u0, iterator up0);

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
    iterator u;
    iterator up;
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

    std::vector<int> step_orders_;
    std::vector<double> step_sizes_;
    static const int variables_per_node = VariableTraits<value_type>::number;

    void copy_vector(N_Vector y, iterator w);

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

/*
template<class Physics, class Preconditioner>
IDAIntegrator<Physics, Preconditioner>::
IDAIntegrator(const Mesh& mesh, double rtol, double atol)
    : m(mesh), pc(), t(), ida_mem(), rtol(rtol), atol(atol) {}

template<class Physics, class Preconditioner>
IDAIntegrator<Physics, Preconditioner>::
IDAIntegrator(const Mesh& mesh, Preconditioner& pc, double rtol, double atol)
    : m(mesh), pc(&pc), t(), ida_mem(), rtol(rtol), atol(atol) {}
*/
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
initialise(double& tt, iterator y, iterator yp, Callback callback)
{
    *procinfo << "\tIDAIntegrator<Physics, Preconditioner>::initialise()" << std::endl;
    t = &tt;
    u = y;
    up = yp;
    compute_residual = callback;

    // Initialise solution vector
    ulocal = N_VNew_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(ulocal);
    value_type* uvec = reinterpret_cast<value_type*>(NV_DATA_P(ulocal));
    // DEVICE
    // need to use minlin to make this transfer
    // and others like it below
    std::copy(&u[0], &u[0] + mesh().local_nodes(), uvec);

    // Initialise derivative vector
    uplocal = N_VNew_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(uplocal);
    value_type* upvec = reinterpret_cast<value_type*>(NV_DATA_P(uplocal));
    std::copy(&up[0], &up[0] + mesh().local_nodes(), upvec);

    // Initialise interpolation vectors used when interpolation is performed
    // on output from IDA, so as not to overwrite previously calculated solutions
    uinterp = N_VNew_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(ulocal);
    upinterp = N_VNew_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(uplocal);

    // Initialise weights vector
    // DEVICE
    // we could crate minlin vectors based on a communicator,
    // then declare weights as N_VNewEmpty_Parrallel
    // and attach it to the minlin vector using
    // N_VSetArrayPointer_Parallel(minlin_vec.data(), weights)
    // and do similarly for the vectors below
    weights = N_VNew_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(weights);

    // Initialise absolute tolerances vector
    atolv = N_VNew_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(atolv);
    std::fill_n(
        NV_DATA_P(atolv),
        mesh().local_nodes() * variables_per_node,
        atol);

    variableids = N_VNew_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(variableids);

    // Create IDA data structure
    ida_mem = IDACreate();
    assert(ida_mem);

    // Initialise IDA internal memory
    int flag = IDAInit(
        ida_mem,
        reinterpret_cast<IDAResFn>(f),   // [1]
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
            reinterpret_cast<IDASpilsPrecSetupFn>(psetup),   // [1]
            reinterpret_cast<IDASpilsPrecSolveFn>(psolve)    // [1]
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
    // DEVICE
    // the interface could be changed so that u and up
    // are references to minlin vectors
    physics.preprocess_timestep( *t, m, u, up );
    int flag = IDASolve(
        ida_mem, 1.0, t, ulocal, uplocal, IDA_ONE_STEP);
    assert(flag == IDA_SUCCESS);
    if( procinfo->rank()==0 )
        std::cout << ".";
    // save the order and size of last step just completed
    int order_last;
    flag = IDAGetLastOrder(ida_mem, &order_last);
    step_orders_.push_back(order_last);
    double step_last;
    flag = IDAGetLastStep(ida_mem, &step_last);
    step_sizes_.push_back(step_last);

    // DEVICE
    // if we use minlin vectors that are attached to empty NVector types we
    // don't have to make copies like this because IDA would make changes
    // to the vector itself.
    copy_vector(ulocal, u);
    copy_vector(uplocal, up);
}

// Advances solution to the specified time
template<class Physics, class Preconditioner>
void IDAIntegrator<Physics, Preconditioner>::advance(double next_time) {

    /*
    value_type* v = reinterpret_cast<value_type*>(NV_DATA_P(ulocal));
    std::copy(&v[0], &v[0]+mesh().local_nodes(), &u[0]);
    v = reinterpret_cast<value_type*>(NV_DATA_P(uplocal));
    std::copy(&v[0], &v[0]+mesh().local_nodes(), &up[0]);
    */
    copy_vector(ulocal, u);
    copy_vector(uplocal, up);

    // advance the solution to next_time
    while( (*t)<next_time )
        advance();

    // interpolate the solution backwards
    int flag = IDAGetDky( ida_mem, next_time, 0, uinterp );
    assert(flag == IDA_SUCCESS);
    flag = IDAGetDky( ida_mem, next_time, 1, upinterp );
    assert(flag == IDA_SUCCESS);

    // copy solution for external reference
    copy_vector(uinterp, u);
    copy_vector(upinterp, up);
}

template<class Physics, class Preconditioner>
void IDAIntegrator<Physics, Preconditioner>::
copy_vector(N_Vector y, iterator w) {
    assert(NV_GLOBLENGTH_P(y) == mesh().global_nodes() * variables_per_node);
    assert(NV_LOCLENGTH_P(y)  == mesh().local_nodes()  * variables_per_node);
    std::copy(
        reinterpret_cast<value_type*>(NV_DATA_P(y)),
        reinterpret_cast<value_type*>(NV_DATA_P(y)) + mesh().local_nodes(),
        w
    );
}

// Returns the absolute tolerance
template<class Physics, class Preconditioner>
typename IDAIntegrator<Physics, Preconditioner>::value_type&
IDAIntegrator<Physics, Preconditioner>::abstol(int i)
{
    assert(ida_mem);
    assert(i >= 0 && i < mesh().local_nodes());
    value_type* avec = reinterpret_cast<value_type*>(NV_DATA_P(atolv));
    return avec[i];
}

template<class Physics, class Preconditioner>
typename IDAIntegrator<Physics, Preconditioner>::value_type
IDAIntegrator<Physics, Preconditioner>::abstol(int i) const
{
    assert(ida_mem);
    assert(i >= 0 && i < mesh().local_nodes());
    const value_type* avec = reinterpret_cast<const value_type*>(NV_DATA_P(atolv));
    return avec[i];
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
void IDAIntegrator<Physics, Preconditioner>::set_algebraic_variables(const std::vector<double> &vals){
    // sanity check the input
    assert(vals.size()==variables_per_node*m.local_nodes());
    for(int i=0; i<vals.size(); i++)
        assert(vals[i]==0. || vals[i]==1.);

    // copy user specified variable ids into NV_Vector
    double* dest = reinterpret_cast<double*>(NV_DATA_P(variableids));
    std::copy(vals.begin(), vals.end(), dest);
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
void IDAIntegrator<Physics, Preconditioner>::compute_initial_conditions(iterator u0, iterator up0){
    int icopt = IDA_Y_INIT;
    if(variableids_set_){
        icopt = IDA_YA_YDP_INIT;
    }

    int flag = IDACalcIC(ida_mem, icopt, (*t)+1.);
    assert(flag==IDA_SUCCESS);

    // get the initial conditions
    N_Vector yy0_mod, yp0_mod;
    yy0_mod = N_VNew_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(yy0_mod);
    yp0_mod = N_VNew_Parallel(
        procinfo->communicator(),
        mesh().local_nodes() * variables_per_node,
        mesh().global_nodes() * variables_per_node);
    assert(yp0_mod);

    flag = IDAGetConsistentIC(ida_mem, yy0_mod, yp0_mod);
    // output to file
    /*
    std::ofstream fid("IC.m");
    value_type* temp = reinterpret_cast<value_type*>(NV_DATA_P(yy0_mod));
    // h0
    fid << "h0 = [" << temp[0].h;
    for(int i=1; i<mesh().local_nodes(); i++)
        fid << ", " << temp[i].h;
    fid << "];" << std::endl;
    // c0
    fid << "c0 = [" << temp[0].c;
    for(int i=1; i<mesh().local_nodes(); i++)
        fid << ", " << temp[i].c;
    fid << "];" << std::endl;
    temp = reinterpret_cast<value_type*>(NV_DATA_P(yp0_mod));
    // hp0
    fid << "hp0 = [" << temp[0].h;
    for(int i=1; i<mesh().local_nodes(); i++)
        fid << ", " << temp[i].h;
    fid << "];" << std::endl;
    // cp0
    fid << "cp0 = [" << temp[0].c;
    for(int i=1; i<mesh().local_nodes(); i++)
        fid << ", " << temp[i].c;
    fid << "];" << std::endl;
    // x
    fid << "x = [" << mesh().node(0).point().x;
    for(int i=1; i<mesh().local_nodes(); i++)
        fid << ", " << mesh().node(i).point().x;
    fid << "];" << std::endl;
    // y
    fid << "y = [" << mesh().node(0).point().y;
    for(int i=1; i<mesh().local_nodes(); i++)
        fid << ", " << mesh().node(i).point().y;
    fid << "];" << std::endl;
    fid.close();
    */

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

// (IDA)
// Computes the residual function
template<class Physics, class Preconditioner>
int IDAIntegrator<Physics, Preconditioner>::
f(double t, N_Vector y, N_Vector yp, N_Vector res, void* ip)
{
    IDAIntegrator* integrator = static_cast<IDAIntegrator*>(ip);

    *integrator->t = t;
    integrator->copy_vector(y,  integrator->u);
    integrator->copy_vector(yp, integrator->up);

    bool communicate = true;
    value_type* r = reinterpret_cast<value_type*>(NV_DATA_P(res));
    return integrator->compute_residual(
        make_iterator<value_type>(r, r, r + integrator->m.local_nodes()), communicate);
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

    iterator u = integrator->u;
    iterator up = integrator->up;

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

    const value_type* res = reinterpret_cast<const value_type*>(NV_DATA_P(r));
    const value_type* w = reinterpret_cast<const value_type*>
        (NV_DATA_P(integrator->weights));
    value_type* temp1 = reinterpret_cast<value_type*>(NV_DATA_P(t1));
    value_type* temp2 = reinterpret_cast<value_type*>(NV_DATA_P(t2));
    value_type* temp3 = reinterpret_cast<value_type*>(NV_DATA_P(t3));

    int result = integrator->preconditioner().setup(
        m, tt, c, h,
        make_iterator<const value_type>(res, res, res + m.local_nodes()),
        make_iterator<const value_type>(w, w, w + m.local_nodes()),
        u, up,
//        make_iterator<value_type>(&u[0], &u[0], &u[0] + m.nodes()),
//        make_iterator<value_type>(&up[0], &up[0], &up[0] + m.nodes()),
        make_iterator<value_type>(temp1, temp1, temp1 + m.local_nodes()),
        make_iterator<value_type>(temp2, temp2, temp2 + m.local_nodes()),
        make_iterator<value_type>(temp3, temp3, temp3 + m.local_nodes()),
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

    iterator u = integrator->u;
    iterator up = integrator->up;

    assert(NV_GLOBLENGTH_P(tmp) == m.global_nodes() * variables_per_node);
    assert(NV_LOCLENGTH_P(tmp) == m.local_nodes() * variables_per_node);

    double h;
    int flag = IDAGetCurrentStep(integrator->ida(), &h);
    assert(flag == 0);
    flag = IDAGetErrWeights(integrator->ida(), integrator->weights);
    assert(flag == 0);

    const value_type* res   = reinterpret_cast<const value_type*>(NV_DATA_P(r));
    const value_type* w = reinterpret_cast<const value_type*>
        (NV_DATA_P(integrator->weights));
    value_type* rhs  = reinterpret_cast<value_type*>(NV_DATA_P(rr));
    value_type* z    = reinterpret_cast<value_type*>(NV_DATA_P(zz));
    value_type* temp = reinterpret_cast<value_type*>(NV_DATA_P(tmp));

    std::copy(rhs, rhs + m.local_nodes(), z);

    return integrator->preconditioner().apply(
        m, tt, c, h, delta,
        make_iterator<const value_type>(res, res, res + m.local_nodes()),
        make_iterator<const value_type>(w, w, w + m.local_nodes()),
        make_iterator<const value_type>(rhs, rhs, rhs + m.local_nodes()),
        u, up,
//        make_iterator<value_type>(&u[0], &u[0], &u[0] + m.nodes()),
//        make_iterator<value_type>(&up[0], &up[0], &up[0] + m.nodes()),
        make_iterator<value_type>(z, z, z + m.local_nodes()),
        make_iterator<value_type>(temp, temp, temp + m.local_nodes()),
        integrator->compute_residual
    );
}

// Definition of static member
template<class Physics, class Preconditioner>
const int IDAIntegrator<Physics, Preconditioner>::variables_per_node;

} // end namespace fvm

#endif
