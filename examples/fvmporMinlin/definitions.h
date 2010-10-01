#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <fvm/mesh.h>
#include <util/doublevector.h>
#include <util/intvector.h>
#include <util/mklallocator.h>

#include <mkl_spblas.h>
#include <mkl_service.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <map>

namespace fvmpor {
using util::DoubleVector;

template<typename TypeName>
class TimeSeries{
    enum dataT {step, interp};
public:
    TimeSeries(){}
    TimeSeries( std::string fname )
    {
        // open the file
        std::ifstream infile;
        infile.open(fname.c_str());
        assert(infile);

        // read the data type
        std::string header;
        infile >> header;
        if( !header.compare("STEP") )
            data_type_ = step;
        else if( !header.compare("INTERPOLATED") )
            data_type_ = interp;
        else
            assert(false);

        // find out how many data points to read
        int nt;
        infile >> nt;

        // read the time values
        t_.resize(nt);
        for(int i=0; i<nt; i++)
            infile >> t_[i]; 

        // read the data samples
        int nvals;
        if( data_type_==step )
            nvals = nt-1;
        else
            nvals = nt;

        vals_.resize(nvals);
        for(int i=0; i<nvals; i++)
            infile >> vals_[i]; 
    }

    bool active() const {
        return( t_.size()!=0 );
    }

    TypeName evaluate( double t ) const{
        assert(t_.size());

        // ensure that t lies inside the range of time values
        assert(t>=t_[0] && t<t_[t_.size()-1]);

        // determine the datapoint that corresponds to t
        int i;
        int N=t_.size();
        for(i=0; i<N-1; i++){
            if( t<t_[i+1] ){
                break;
            }
        }

        TypeName val = TypeName();
        double w;
        switch(data_type_){
            case interp:
                w = (t-t_[i])/(t_[i+1]-t_[i]);
                val = w*vals_[i+1] + (1.-w)*vals_[i];
                break;
            case step:
                val = vals_[i];
                break;
        }

        return val;
    }
private:
    dataT data_type_;
    std::vector<double> t_;
    std::vector<TypeName> vals_;
};

class BoundaryCondition{
public:
    BoundaryCondition(){
        type_ = -1;
        value_ = 0.;
        b_ = 0.;
        eta_ = 0.;
        q_ = mesh::Point();
    }
    static BoundaryCondition Dirichlet( double val ){
        return BoundaryCondition(1, val);
    }
    static BoundaryCondition PrescribedFlux( double val ){
        return BoundaryCondition(3, val);
    }
    static BoundaryCondition Dirichlet( std::string fname ){
        return BoundaryCondition(1, fname);
    }
    static BoundaryCondition PrescribedFlux( std::string fname ){
        return BoundaryCondition(3, fname);
    }
    static BoundaryCondition Hydrostatic( double b, double eta ){
        return BoundaryCondition(4, b, eta);
    }
    static BoundaryCondition Hydrostatic( std::string fname, double eta ){
        return BoundaryCondition(4, fname, eta);
    }
    static BoundaryCondition PrescribedDirectionalFlux( mesh::Point val ){
        return BoundaryCondition(6, val);
    }
    static BoundaryCondition PrescribedDirectionalFlux( std::string fname ){
        return BoundaryCondition(6, fname);
    }
    static BoundaryCondition Seepage(){
        return BoundaryCondition(7);
    }
    static BoundaryCondition HydrostaticShore( double b, double eta ){
        return BoundaryCondition(8, b, eta);
    }
    static BoundaryCondition HydrostaticShore( std::string fname, double eta ){
        return BoundaryCondition(8, fname, eta);
    }

    // returns the type of the BC
    int type() const{
        return type_;
    }

    // returns whether the BC is dirichlet
    bool is_dirichlet() const{
        if( type_==1 || type_==4 )
            return true;
        return false;
    }

    // return value_
    double value(double t) const{
        assert(type_==1 || type_==3 || type_==7); // only applys for dirichlet and fixed flux
        if( scalar_time_series_data.active() )
            return scalar_time_series_data.evaluate(t);
        return value_;
    }

    // return directional flux relative to a surface normal n
    double flux( double t, const mesh::Point& n ) const{
        assert(type_==6 || type_==7);
        if( type_==6 ){
            if( vector_time_series_data.active() ){
                mesh::Point val = vector_time_series_data.evaluate(t);
                return dot(n,val);
            }
        }
        return dot( n, q_ );
    }

    // return hydrostatic pressure at elevation z
    double hydrostatic( double t, double z ) const{
        assert(type_==4);
        if( scalar_time_series_data.active() )
        {
            double val = (1.+eta_)*(scalar_time_series_data.evaluate(t)-z);
            return val;
        } else{
            double val = (1.+eta_)*(b_-z);
            return val;
        }
    }

    // return hydrostatic pressure at elevation z
    double hydrostatic_shore( double t, double z ) const{
        assert(type_==8);
        if( scalar_time_series_data.active() )
        {
            double val = (1.+eta_)*(scalar_time_series_data.evaluate(t)-z);
            return val;
        } else{
            double val = (1.+eta_)*(b_-z);
            return val;
        }
    }
private:

    BoundaryCondition( int type, double val ){
        assert( type==1 || type==3 ); // ensure either dirichlet or prescribed flux
        type_ = type;
        value_ = val;
    }

    // hydrostatic boundary : pressure head = b - (1+eps)(b-depth)
    BoundaryCondition( int type, double b, double eta ){
        assert( type==4 || type==8 );
        b_ = b;
        type_ = type;
        eta_ = eta;
    }
    // hydrostatic boundary : pressure head = b - (1+eps)(b-depth)
    // where b is takend from an input file
    BoundaryCondition( int type, std::string fname, double eta ){
        // ensure that the user has requested a hydrostatic condition
        assert( type==4 || type==8 );

        scalar_time_series_data = TimeSeries<double>(fname);

        type_ = type;
        eta_ = eta;
    }

    // directional flux boundary
    BoundaryCondition( int type, const mesh::Point p ){
        assert( type==6 );
        type_ = type;
        q_ = p;
    }

    // Boundary condition is specified by file
    BoundaryCondition( int type, std::string fname ){
        assert(type==6 || type==1 || type ==3);
        type_ = type;
        if( type==6 )
            vector_time_series_data = TimeSeries<mesh::Point>(fname);
        else
            scalar_time_series_data = TimeSeries<double>(fname);
    }

    // seepage boundary
    BoundaryCondition( int type ){
        assert( type==7 );
        type_ = 7;
        value_ = 0.;
        // seepage boundary has dirichlet head fixed at zero in saturated region
        // also has flux over boundary of zero in the unsaturated region
    }

    int type_;
    double value_;
    double b_;
    double eta_;
    mesh::Point q_;
    TimeSeries<double> scalar_time_series_data;
    TimeSeries<mesh::Point> vector_time_series_data;
};

struct PhysicalZone {
    // permeability
    double K_xx;
    double K_yy;
    double K_zz;
    // porosity
    double phi;
    // van Ganuchten-Mualem parameters
    double alphaVG;
    double nVG;
    double S_r;
    // soil compression
    double alpha;

    /////////////////////
    // DERIVED
    /////////////////////
    double S_op;
    double mVG;
    double alpha_h;

    friend std::ostream& operator<<(std::ostream& os,
                                    const PhysicalZone& val)
    {
        os << "Kxx, Kyy, Kzz : " << val.K_xx << ' ' << val.K_yy << ' ' << val.K_zz << std::endl;
        os << "porosity : " << val.phi << std::endl;
        os << "alphaVG, nVG, mVG : " << val.alphaVG << ' ' << val.nVG << ' ' << val.mVG <<  std::endl;
        os << "S_r : " << val.S_r << std::endl;
        os << "S_op : " << val.S_op << std::endl;
        os << "alpha, alpha_h : " << val.alpha << ' ' << val.alpha_h << std::endl;
        return os;
    }
};

class Constants{
public:
    double mu() const {return mu_;}; // viscosity
    double beta() const {return beta_;}; // fluid compressibility
    double g() const {return g_;};
    double rho_0() const {return rho_0_;};

    Constants( double mu, double beta, double g, double rho_0 ):
        mu_(mu), beta_(beta), g_(g), rho_0_(rho_0) {
    }
    // defaults use metres, seconds, kg
    Constants() : mu_(1e-3), beta_(4.47e-10), g_(9.80665), rho_0_(1000.0) {
    }
    friend std::ostream& operator<<(std::ostream& os, const Constants& c){
        os << "beta = " << c.beta() << std::endl;
        os << "mu = " << c.mu() << std::endl;
        os << "g = " << c.g() << std::endl;
        os << "rho_0 = " << c.rho_0() << std::endl;
        return os;
    }
private:
    double mu_; // viscosity
    double beta_; // fluid compressibility
    double g_; // gravity
    double rho_0_; // fresh water density
};

}

#endif
