//
//  boundary and physical conditions for cassion problem
#include "fvmpor.h"

namespace fvmpor{

    // initial conditions
    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::set_initial_conditions( double &t, const mesh::Mesh& m ){
        spatial_weighting = weightUpwind;
        //spatial_weighting = weightAveraging;
        //spatial_weighting = weightVanLeer;

        if( true ){
            std::ifstream fid("ic.txt");
            std::vector<double> hh(m.local_nodes());
            std::vector<double> cc(m.local_nodes());
            for (int i = 0; i < m.local_nodes(); ++i){
                fid >> h_vec_[i];
                c_vec_[i] = 0.;
            }
        }else{
            for (int i = 0; i < m.local_nodes(); ++i) {
                const mesh::Node& n = m.node(i);
                Point p = n.point();
                double x = p.x;
                double el = dimension == 2 ? p.y : p.z;

                if( is_dirichlet_h_vec_[i] )
                {
                    int tag = is_dirichlet_h_vec_[i];
                    if( boundary_condition_h(tag).type()==1 )
                        h_vec_[i] = boundary_condition_h(tag).value(t);
                    else
                        h_vec_[i] = boundary_condition_h(tag).hydrostatic(t,el);
                } else{
                    // due to head being an alegbraic variable, this value will be change
                    // when consistent initial conditions are computed
                    h_vec_[i] = 100.-el;
                    //h_vec_[i] = 0.;
                }
                h_vec_[i] = 0.;
                if( is_dirichlet_c_vec_[i] ){
                    int tag = is_dirichlet_c_vec_[i];
                    assert( boundary_condition_c(tag).type()==1);
                    c_vec_[i] = boundary_condition_c(tag).value(t);
                } else{
                    c_vec_[i] = 0.;
                }
            }
        }
    }

    // set physical zones
    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::set_physical_zones( void )
    {
        double rho_0 = constants().rho_0();
        double g =  constants().g();
        double mu = constants().mu();

        PhysicalZone zone;

        double factor = 1./(24.*3600.);
        zone.K_xx = zone.K_yy = zone.K_zz = 1.*factor;
        zone.phi = 0.35;
        zone.Dm = factor*6.6e-2;
        //zone.Dm = 6.6e-4;

        physical_zones_.push_back(zone);
    }

    // set constants for simulation
    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::set_constants(){
        constants_ = Constants(1e-3, 0.025, 9.80665, 1000.0);
        //constants_ = Constants(1e-3, 0., 9.80665, 1000.0);
    }

    // boundary conditions
    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::set_boundary_conditions(){
        // no flow boundaries
        boundary_conditions_h_[1] = BoundaryCondition::PrescribedFlux(0.);
        boundary_conditions_c_[1] = BoundaryCondition::PrescribedFlux(0.);

        // Dirichlet on RHS boundary
        //boundary_conditions_h_[2] = BoundaryCondition::Hydrostatic(100., .025);
        boundary_conditions_h_[2] = BoundaryCondition::Hydrostatic(100., .025);
        boundary_conditions_c_[2] = BoundaryCondition::ASE(1.);

        // inflow on LHS boundary
        boundary_conditions_h_[3] = BoundaryCondition::PrescribedFlux(-7.39e-8); // TRUE HENRY BC VALUE
        //boundary_conditions_c_[3] = BoundaryCondition::PrescribedFlux(-2.39e-8);
        boundary_conditions_c_[3] = BoundaryCondition::ASE(0.);
    }
}

