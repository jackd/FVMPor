#ifndef EEM_INTEGRATOR_H
#define EEM_INTEGRATOR_H

#include <fvm/fvm.h>
#include <fvm/mesh.h>
#include <mpi/mpicomm.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include <math.h>

#include "missing_lin.h"

namespace fvm {

template<class Physics>
class EEMIntegrator {
public:
	typedef typename Physics::TVecDevice TVecDevice;
	typedef typename Physics::TVec TVec;
	typedef typename lin::Matrix<double> DMatrix;
	typedef typename lin::Vector<double> DVector;
	
    typedef mesh::Mesh Mesh;

    typedef typename Physics::value_type value_type;
    typedef typename fvm::Callback<Physics> Callback;

    EEMIntegrator(const Mesh& mesh, Physics& ph, double rtol, double atol);

    void initialise(double& t, TVecDevice &u, TVecDevice &up, Callback compute_residual);

    const Mesh& mesh() const;

    void advance();                 // by one internal timestep
    void advance(double next_time); // to specified time

    // Returns the absolute tolerance
    double& abstol();
    double abstol() const;

    // Returns the relative tolerance
    double& reltol();
    double reltol() const;

	// **REDUNDANT** I think
    // Sets integration tolerances
    void set_tolerances();

    // set the maximum timestep taken by EEMSolve
    void set_max_timestep(double);
    void set_max_order(int); 							// **REDUNDANT**
    double max_timestep() const;
    int max_order() const;								// **REDUNDANT** (ish)
    void set_algebraic_variables(const TVec &vals);		// **REDUNDANT**
    void compute_initial_conditions(TVecDevice &u0, TVecDevice &up0);

    // return pointer to the step orders
    const std::vector<int>& step_orders() const{
        return step_orders_;							// **REDUNDANT** (ish)
    }

    // return pointer to the step sizes
    const std::vector<double>& step_sizes() const{
        return step_sizes_;
    }

    // Will keep for the moment. Doesn't do anything mind you.
    void* ida();										// **REDUNDANT**
	
	typename Physics::TVecDevice ur() const;				//return u
    
private:
    const Mesh& m;				//mesh over which to solve
    Physics& physics;			//problem description
    mpi::MPICommPtr procinfo;
    Callback compute_residual;	//function computing G(u) = dudt **FIXME**
    double t;					//time at most recent solution recorded (ulocal)
	double tau_last;			//step size of most recent successful step
    double rtol;				//relative tolerance used by wrms_norm
    double atol;				//absolute tolerance used by wrms_norm
    double max_timestep_;		//maximum allowable timestep
    int max_order_;				// **REDUNDANT**
    bool variableids_set_;		// **REDUNDANT**
    TVecDevice u;				//solution at latest saved time
    TVecDevice ulocal;			//solution at latest calculated time
	TVecDevice Glocal;			//G(ulocal,t)
	double beta;				//beta is norm(Glocal)
	
	void copy_data(DVector& target, const DVector& source);

    std::vector<int> step_orders_; // **REDUNDANT**
    std::vector<double> step_sizes_; //saved values of tau
    static const int variables_per_node = VariableTraits<value_type>::number;
	
	double tau;
	double tau_min;
	
	void print_stats();
	int EEMSolve();
	int step_in_time(TVecDevice& unew, bool step_twice);
	double wrms_norm(const TVecDevice& x);
	typename Physics::TVecDevice G(const TVecDevice &umod, double t);
	
	int tstep_success;
	int tstep_failure_kry;
	int tstep_failure_inc;
	
	double eta;
	double eta_bar;
	int jmax;			//maximum allowable krylov subspace dimension
};

template<class Physics>
typename Physics::TVecDevice EEMIntegrator<Physics>::ur() const{
	return u;
}

template<class Physics>
EEMIntegrator<Physics>::
EEMIntegrator(const Mesh& mesh, Physics& physics, double rtol, double atol)
    : m(mesh), physics(physics), t(), rtol(rtol), atol(atol), max_timestep_(0.), max_order_(5), jmax(10), variableids_set_(false), eta(0.9), eta_bar(0.25) {
    procinfo = m.mpicomm()->duplicate("EEM");
}

template<class Physics>
void EEMIntegrator<Physics>::
initialise(double& tt, TVecDevice &y, TVecDevice &yp, Callback callback)
{
    *procinfo << "\tEEMIntegrator<Physics>::initialise()" << std::endl;
    t = tt;

    int localSize = mesh().local_nodes()*variables_per_node;
    int globalSize = mesh().global_nodes()*variables_per_node;
	
	tstep_success = 0;
	tstep_failure_kry = 0;
	tstep_failure_inc = 0;
	
	tau_min = 0.;

    u = TVecDevice(mesh().nodes()*variables_per_node, y.data());
	ulocal = u;
	//ulocal = y;
	//ulocal = TVecDevice(localSize,y.data());
	//ulocal = TVecDevice(localSize,y.data());
    compute_residual = callback;
	tau = 1.0;
}

template<class Physics>
const mesh::Mesh& EEMIntegrator<Physics>::mesh() const {
    return m;
}

// Advances solution by one internal timestep
template<class Physics>
void EEMIntegrator<Physics>::advance() {
    // we copy ulocal into u because the ulocal contains the current
    // solution value inside IDA, whereas u and up may contain a version
    // of the solution that was interpolated backwards
    // make this copy to ensure that up to date values are used for calculating
    // preprocess_timestep()
    //u = ulocal;
	copy_data(u,ulocal);
    physics.preprocess_timestep( t, m,u, u );

	Glocal = G(ulocal,t);
	beta = norm(Glocal);
	
    int flag = EEMSolve();
	if( procinfo->rank()==0 )
		std::cerr << ".";

	step_orders_.push_back(2);
	step_sizes_.push_back(tau_last);
	t += tau_last;

}

// Advances solution to the specified time
template<class Physics>
void EEMIntegrator<Physics>::advance(double next_time) {

    // advance the solution to next_time
    while( t < next_time )
        advance();

    // interpolate the solution backwards to next_time
    u += ((next_time - (t - tau_last))/tau_last)*(ulocal - u);
	//std::cerr << "max(u) = " << max(u) << std::endl;
}

// Returns the absolute tolerance
template<class Physics>double& EEMIntegrator<Physics>::abstol()
{
    return &atol;
}

template<class Physics>
double EEMIntegrator<Physics>::abstol() const
{
    return atol;
}

// Returns the relative tolerance
template<class Physics>
double& EEMIntegrator<Physics>::reltol() {
    return rtol;
}

template<class Physics>
double EEMIntegrator<Physics>::reltol() const {
    return rtol;
}

// Sets integration tolerances
template<class Physics>
void EEMIntegrator<Physics>::set_tolerances() {
	std::cerr << "Attempted to set IDA tolerance, but using EEM integrator" << std::endl;
}

template<class Physics>
void EEMIntegrator<Physics>::set_algebraic_variables(const TVec &vals){
    std::cerr << "Attempted to set algebraic variables. EEM method requires ODE, not DAE" << std::endl;
}

template<class Physics>
void EEMIntegrator<Physics>::compute_initial_conditions(TVecDevice &u0, TVecDevice &up0){
    std::cerr << "Attempted to compute consistent initial conditions. EEM method does not requrie this" << std::endl;
}

// set the maximum timestep taken by EEM
template<class Physics>
void EEMIntegrator<Physics>::set_max_timestep(double max_ts) {
    max_timestep_ = max_ts;
}

template<class Physics>
void EEMIntegrator<Physics>::set_max_order(int max_order) {
    std::cerr << "Attempted to set max order, but order of EEM method always 2" << std::endl;
}

template<class Physics>
void* EEMIntegrator<Physics>::ida() {
	std::cerr << "Attempted to access ida data structure whilst using EEM integrator" << std::endl;
    void* temp;
	return temp;
}

// Computes G(u,t)
template<class Physics >
typename Physics::TVecDevice EEMIntegrator<Physics>::G(const TVecDevice &umod, double tmod)
{
	//**FIXME** Update with Ben's stuff
	TVecDevice utemp = u;
	//u = umod;
	copy_data(u,umod);
	double ttemp = t;
	t = tmod;
    TVecDevice r(m.local_nodes()*variables_per_node);
    bool communicate = true;
    int success = compute_residual(r, communicate);
	//u = utemp;
	copy_data(u,utemp);
	t = ttemp;
    return r;

/*
	//TVecDevice utemp = u;
	//u = umod;
	//doublt ttemp = t;
	//t = tmod;
	TVecDevice r(m.local_nodes()*variables_per_node);
	bool communicate = true;
	int success = compute_residual(r,umod,tmod,communicate);
	//u = utemp;
	//t = ttemp;
	return r;
*/
}

template<class Physics>
double EEMIntegrator<Physics>::wrms_norm(const TVecDevice& x){
	double temp = 0.0;
	for (int i = 1; i < x.size() + 1; i++){
		double w = rtol*x(i) + atol;
		temp += x(i)*x(i)/w/w;
	}
	temp /= x.size();
	return std::sqrt(temp);
}

/*Function for taking a step of length tau in time. Step can either be full (step_twice = false)
or two half steps (step_twice = true).*/
template<class Physics>
int EEMIntegrator<Physics>::step_in_time(TVecDevice& unew, bool step_twice){
	double termination_val = 1.0;
	double tau_used = tau;
	if (step_twice){
		tau_used /= 2;
		termination_val = 0.5;
	}
	int n = unew.size();
    DMatrix V(n, jmax+1);
    DMatrix H(jmax+1, jmax);
    
	if (beta == 0.0){
		unew = ulocal;
		std::cout << "Glocal zero, implies dudt = 0" << std::endl;
		return 0;
	}
	//V(lin::all,1) = Glocal/beta; //**FIXME**
	for (int i = 1; i <= n; ++i){
		V(i,1) = Glocal(i)/beta;
	}

    int j = 0;
	double epsm = std::numeric_limits<double>::epsilon();
	DMatrix phiH; //**FIXME** TMatDevice phiH(jmax,jmax) would be better I think, gets tricky though
	double epsilon = std::sqrt(epsm)*norm(ulocal);
	int failed = 0;
	if (epsilon == 0){
	//**FIXME** Not sure what to do in this situation...
		epsilon = sqrt(epsm);
		std::cout << "Possible source of error - norm(ulocal) = 0" << std::endl;
	}
	while (true){
		if (j >= jmax){
			failed = 1;
			break;
		}
        ++j;
		//TVecDevice w = (G(ulocal + epsilon*V(lin::all,j),t) + (-1)*Glocal)/epsilon; //**FIXME**
		TVecDevice w = plus(G(plus(ulocal,mul(epsilon,subv(V,1,n,j))),t),mul(-1,Glocal))*(1.0/epsilon);
		for (int i = 1; i <= j; ++i) {
            //H(i,j) = dot(V(lin::all,i), w);//**FIXME**
			H(i,j) = dot(subv(V,1,n,i),w);
            //w -= H(i,j)*V(lin::all,i);//**FIXME**
			w = plus(w,mul(-H(i,j),subv(V,1,n,i)));
        }

        for (int i = 1; i <= j; ++i) {
            //double c = dot(V(lin::all,i), w);//**FIXME**
			double c = dot(subv(V,1,n,i),w);
            H(i,j) += c;
            //w -= c * V(lin::all,i);//**FIXME**
			w = plus(w,mul(-c,subv(V,1,n,i)));
        }
    
        H(j+1,j) = norm(w);
		//V(lin::all,j+1) = w/H(j+1,j); //**FIXME**
		for (int i = 1; i <= n; ++i){
			V(i,j+1) = w(i)/H(j+1,j);
		}
		
		//phiH = phipade(tau_used*H(1,j,1,j));//**FIXME
		phiH = phipade(mul(tau_used,subm(H,1,j,1,j)));
        if (H(j+1,j) <= n*n*epsm) { 
            std::cerr << "Broke down. j = \t" << j << std::endl;
            break;
        }
		else{
			//TVecDevice rho = (tau_used*beta*H(j+1,j)*phiH(j,1))*V(lin::all,j+1); //**FIXME**
			TVecDevice rho = mul(tau_used*beta*H(j+1,j)*phiH(j,1),subv(V,1,n,j+1));
			if (wrms_norm(rho)*tau_used < termination_val)
			{
				break;
			}
        }
    }
	//unew = ulocal + (tau_used*beta)*(V(lin::all,1,j)*phiH(lin::all,1));  //**FIXME**
	unew = plus(ulocal, mul(tau_used*beta,mul(subm(V,1,n,1,j),subv(phiH,1,j,1))));
	if (step_twice){
		TVecDevice G_half = G(unew,t);
		//unew += tau_used*(V(lin::all,1,j)*(phiH*(transpose(V(lin::all,1,j))*G_half))); //**FIXME**
		unew = plus(unew,mul(tau_used,mul(subm(V,1,n,1,j),mul(phiH,mul(transpose(subm(V,1,n,1,j)),G_half)))));
	}
	return failed;
}

/*Function for taking a step of length tau twice (once with a full step, once with two tau/2 steps).
Assuming both succeed, the solutions are compared. If they are suitably close, the full step solution
is adopted. */
template<class Physics>
int EEMIntegrator<Physics>::EEMSolve(){
	int n = ulocal.size();
    
    bool failed_full = 0;
    bool failed_half = 0;
	double ndu = 0;
    
    int success = 1;
    TVecDevice u_full(n);
    TVecDevice u_half(n);
	if (beta == 0){
		success = 0;
		u_full = ulocal;
	}
	while(success != 0){
		success = 0;
		failed_full = step_in_time(u_full, 0); //calculate the normal way
		if (failed_full){
			//std::cout << "failed_full" << std::endl;
			success = 1;
			++(tstep_failure_kry);
			if(tau < tau_min){
				std::cerr << "failed with tau < tau_min" << std::endl;
			}
			tau /= 2;
		}
		else{
			failed_half = step_in_time(u_half,1);//calculate using 2 half-steps
			if (failed_half){
				//std::cout << "failed_half" << std::endl;
				success = 2;
				++(tstep_failure_kry);
				if(tau < tau_min){
					std::cerr << "failed with tau < tau_min" << std::endl;
				}
				tau /= 2;
			}
			else{
				TVecDevice du = u_full + (-1)*u_half;	//compare difference
				ndu = wrms_norm(du);
				if (ndu*eta_bar >= 1){
					//std::cout << "failed_inc" << std::endl;
					success = 3;
					++(tstep_failure_inc);
					if(tau < tau_min){
						std::cerr << "failed with tau < tau_min" << std::endl;
					}
					tau /= 2;
				}
			}
		}
	}
	//std::cout << "Advance successful. Biggest change: \t" << max(abs(ulocal - u_full)) << std::endl;
	ulocal = u_full;
	tau_last = tau;
	tau *= min(eta*std::pow(1/ndu,1.0/3.0),2.0);
	success = 0;
	++(tstep_success);
	return success;
}

template<class Physics>
void EEMIntegrator<Physics>::print_stats()
{
	std::cout << "time solved to: \t \t" << t << std::endl;
    std::cout << "tstep_success: \t \t" << tstep_success << std::endl;
    std::cout << "tstep_failure_kry: \t" << tstep_failure_kry << std::endl;
    std::cout << "tstep_failure_inc: \t" << tstep_failure_inc << std::endl;
}

template<class Physics>
void EEMIntegrator<Physics>::copy_data(DVector& target, const DVector& source){
	assert(target.size() == source.size());
	for (int i = 1; i <= target.size();++i){
		target(i) = source(i);
	}
}

// Definition of static member
template<class Physics>
const int EEMIntegrator<Physics>::variables_per_node;

} // end namespace fvm

#endif
