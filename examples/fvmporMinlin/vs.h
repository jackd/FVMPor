//  boundary and physical conditions for cassion problem
#include "fvmpor.h"

namespace fvmpor{

// initial conditions
template <typename CoordLocal, typename CoordCompute>
void VarSatPhysicsImpl<CoordLocal,CoordCompute>::set_initial_conditions( double &t, const mesh::Mesh& m ){
    spatial_weighting = weightAveraging; // of weightUpwind, weightAveraging, weightVanLeer
    //spatial_weighting = weightUpwind; // of weightUpwind, weightAveraging, weightVanLeer

    for (int i = 0; i < m.local_nodes(); ++i) {
        const mesh::Node& n = m.node(i);
        Point p = n.point();
        double x = p.x;
        double el = dimension == 2 ? p.y : p.z;
    
        //h_vec[i] = -1.;
        h_vec[i] = -(el+1.);
        if( is_dirichlet_h_vec_[i] ){
            int tag = is_dirichlet_h_vec_[i];
            int type = boundary_condition_h(tag).type();
            // dirichlet
            if( type==1 ) 
                h_vec.at(i) = boundary_condition_h(tag).value(t);
            // dirichlet hydrostatic
            else if( type==4 )
                h_vec.at(i) = boundary_condition_h(tag).hydrostatic(t, el);
        }
    }
}

// set physical zones
template <typename CoordLocal, typename CoordCompute>
void VarSatPhysicsImpl<CoordLocal,CoordCompute>::set_physical_zones( void )
{
    double rho_0 = constants().rho_0();
    double g =  constants().g();
    double beta = constants().beta();
    double mu = constants().mu();
    //double alpha = 1e-8;
    double alpha = 0.;

    PhysicalZone zone;

    // zone 0
    zone.K_xx = zone.K_yy = zone.K_zz = 9.2e-5;
    zone.phi = .368;
    zone.alphaVG = 3.55;
    zone.alpha = alpha;
    zone.nVG = 2;
    zone.mVG = 1.- 1./zone.nVG;
    zone.S_r = 0.3261;
    zone.alpha_h = zone.alpha * rho_0 * g * (1.0-zone.phi) / zone.phi;
    zone.S_op = ((1.0-zone.phi) * zone.alpha + zone.phi * beta);

    physical_zones_.push_back(zone);

    /*
    double KFactor = 1.e10;
    zone.K_xx = zone.K_xx*KFactor;
    zone.K_zz = zone.K_zz*KFactor;
    physical_zones_.push_back(zone);
    */
}

// set constants for simulation
template <typename CoordLocal, typename CoordCompute>
void VarSatPhysicsImpl<CoordLocal,CoordCompute>::set_constants(){
    constants_ = Constants(1e-3, 0., 9.80665, 1000.0);
}

// boundary conditions
template <typename CoordLocal, typename CoordCompute>
void VarSatPhysicsImpl<CoordLocal,CoordCompute>::set_boundary_conditions(){
    // no flow boundaries
    boundary_conditions_h_[1] = BoundaryCondition::PrescribedFlux(0.);

    // inflow on top LHS boundary
    boundary_conditions_h_[2] = BoundaryCondition::PrescribedFlux(-2.e-5);

    // rhs boundary
    boundary_conditions_h_[3] = BoundaryCondition::PrescribedFlux(0.);
}

}
