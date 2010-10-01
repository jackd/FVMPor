//
//  boundary and physical conditions for cassion problem
#include "fvmpor.h"

namespace fvmpor{

    // initial conditions
    template <typename TVec, typename TIndexVec>
    void VarSatPhysicsImpl<TVec, TIndexVec>::set_initial_conditions( double &t, const mesh::Mesh& m ){
        spatial_weighting = weightUpwind;
        //spatial_weighting = weightAveraging;
        //spatial_weighting = weightVanLeer;

        for (int i = 0; i < m.local_nodes(); ++i) {
            const mesh::Node& n = m.node(i);
            Point p = n.point();
            double x = p.x;
            double el = dimension == 2 ? p.y : p.z;

            if( is_dirichlet_h_vec_[i] ){
                int tag = is_dirichlet_h_vec_[i];
                if( boundary_condition_h(tag).type()==1 )
                    h_vec[i] = boundary_condition_h(tag).value(t);
                else
                    h_vec[i] = boundary_condition_h(tag).hydrostatic(t,el);
            } else{
                h_vec[i] = -7.34;
                //h_vec[i] = -100.;
            }
        }
    }

    // set physical zones
    template <typename TVec, typename TIndexVec>
    void VarSatPhysicsImpl<TVec, TIndexVec>::set_physical_zones( void )
    {
        double rho_0 = constants().rho_0();
        double g =  constants().g();
        double beta = constants().beta();
        double mu = constants().mu();
        //double alpha = 1e-8;
        double alpha = 0.;

        PhysicalZone zone;

        // LARGE CASSION PROBLEM
        double factor = rho_0*g/mu;
        // top layer : zone 1
        zone.K_xx = zone.K_yy = zone.K_zz = factor*9.33e-12;
        zone.phi = 0.368;
        zone.alphaVG = 3.34;
        zone.alpha = alpha;
        zone.nVG = 1.982;
        zone.mVG = 1.- 1./zone.nVG;
        zone.S_r = 0.2771;
        zone.alpha_h = zone.alpha * rho_0 * g * (1.0-zone.phi) / zone.phi;
        zone.S_op = ((1.0-zone.phi) * zone.alpha + zone.phi * beta);

        physical_zones_.push_back(zone);

        // second to top layer : zone 2
        zone.K_xx = zone.K_yy = zone.K_zz = factor*5.55e-12;
        zone.phi = 0.3510;
        zone.alphaVG = 3.63;
        zone.alpha = alpha;
        zone.nVG = 1.632;
        zone.mVG = 1.- 1./zone.nVG;
        zone.S_r = 0.2806;
        zone.alpha_h = zone.alpha * rho_0 * g * (1.0-zone.phi) / zone.phi;
        zone.S_op = ((1.0-zone.phi) * zone.alpha + zone.phi * beta);
        physical_zones_.push_back(zone);

        // Bottom region : zone 3
        zone.K_xx = zone.K_yy = zone.K_zz = factor*4.898e-12;
        zone.phi = 0.3250;
        zone.alphaVG = 3.45;
        zone.alpha = alpha;
        zone.nVG = 1.573;
        zone.mVG = 1.- 1./zone.nVG;
        zone.S_r = 0.2643;
        zone.alpha_h = zone.alpha * rho_0 * g * (1.0-zone.phi) / zone.phi;
        zone.S_op = ((1.0-zone.phi) * zone.alpha + zone.phi * beta);
        physical_zones_.push_back(zone);

        // Central region : zone 4
        zone.K_xx = zone.K_yy = zone.K_zz = factor*4.898e-11;
        zone.phi = 0.3250;
        zone.alphaVG = 3.45;
        zone.alpha = alpha;
        zone.nVG = 1.573;
        zone.mVG = 1.- 1./zone.nVG;
        zone.S_r = 0.2643;
        zone.alpha_h = zone.alpha * rho_0 * g * (1.0-zone.phi) / zone.phi;
        zone.S_op = ((1.0-zone.phi) * zone.alpha + zone.phi * beta);
        physical_zones_.push_back(zone);
    }

    // set constants for simulation
    template <typename TVec, typename TIndexVec>
    void VarSatPhysicsImpl<TVec, TIndexVec>::set_constants(){
        constants_ = Constants(1e-3, 0., 9.80665, 1000.0);
        //constants_ = Constants(1e-3, 1.e-8, 9.80665, 1000.0);
    }

    // boundary conditions
    template <typename TVec, typename TIndexVec>
    void VarSatPhysicsImpl<TVec, TIndexVec>::set_boundary_conditions(){
        // no flow boundaries
        boundary_conditions_h_[1] = BoundaryCondition::PrescribedFlux(0.);

        // inflow on top right hand boundary
        boundary_conditions_h_[2] = BoundaryCondition::PrescribedFlux(-2.3148e-7);
        //boundary_conditions_h_[2] = BoundaryCondition::PrescribedFlux(0.);
    }
}

