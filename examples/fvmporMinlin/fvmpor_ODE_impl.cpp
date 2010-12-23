#include "fvmpor_ODE.h"
#ifdef PROBLEM_CASSION
#include "cassion.h"
#endif
#ifdef PROBLEM_VS
#include "vs.h"
#endif

#include <mkl_vml_functions.h>

#include <util/doublevector.h>

#include <algorithm>
#include <vector>
#include <limits>
#include <utility>

#include <math.h>

// For debugging
#include <iostream>
#include <ostream>
#include <iomanip>
#include <iterator>

namespace fvmpor {

    std::ostream_iterator<Physics::value_type> cit(std::cerr, " ");

    using lin::end;
    using lin::all;

    template<>
    void Physics::initialise(double& t, const mesh::Mesh& m,
                             TVecDevice &u, TVecDevice &udash, TVecDevice &temp,
                             Callback compute_residual)
    {
        // allocate storage
        initialise_vectors( m );

        // Set initial values
        set_initial_conditions(t, m);
        u.at(all) = h_vec;

        // Compute residual
        compute_residual(temp, true);

        // Set initial derivatives
        TVec ahh_vec_host(ahh_vec);
        TVec temp_host(temp);
        TVec udash_host(udash);
        for (int i = 0; i < m.local_nodes(); ++i) {
            if( !is_dirichlet_h_vec_[i] ){
                if( ahh_vec_host[i] )
                    udash_host[i] = temp_host[i]/ahh_vec_host[i];
            }
        }
        udash.at(0,m.local_nodes()-1) = udash_host.at(0,m.local_nodes()-1);
    }

    template<>
    void Physics::preprocess_evaluation(double t, const mesh::Mesh& m,
                                        const TVecDevice &u, const TVecDevice &udash)
    {
        ++num_calls;
        double T;

        // Copy h and hp from the passed iterators
        // though we might be able to avoid this completely
        // and just use the passed references
        h_vec.at(all) = u;
        hp_vec_.at(all) = udash;

        // Compute shape function values and gradients
        shape_matrix.matvec( h_vec, h_faces );
        shape_gradient_matrixX.matvec( h_vec, grad_h_faces_.x() );
        shape_gradient_matrixY.matvec( h_vec, grad_h_faces_.y() );
        if (dimension == 3){
            shape_gradient_matrixZ.matvec( h_vec, grad_h_faces_.z() );
        }

        // determine the p-s-k values
        process_volumes_psk( m );

        // determine derivative coefficients
        process_derivative_coefficients( m );

        // density at faces using shape functions
        process_faces_shape( m );

        // Compute psk values at faces using upwinding/limiting
        process_faces_lim( m );

        // compute fluxes
        process_fluxes( t, m );
    }

    template<>
    void Physics::residual_evaluation( double t, const mesh::Mesh& m,
                                       const TVecDevice &sol, const TVecDevice &deriv,
                                       TVecDevice &res)
    {
        // collect fluxes to CVs
        cvflux_matrix.matvec(M_flux_faces, res);

        // add the source terms here
        //res += source_vec;

        // subtract the lhs
        res -= mul(hp_vec_,ahh_vec);

        // Dirichlet boundary conditions
        res.at(dirichlet_nodes_)  = h_dirichlet_;
        res.at(dirichlet_nodes_) -= h_vec.at(dirichlet_nodes_);
    }

    template<>
    void Physics::preprocess_timestep( double t, const mesh::Mesh& m,
                                       const TVecDevice &sol, const TVecDevice &deriv)
    {
        //--------------------------------
        // determine the spatial weights
        //--------------------------------
        // find upwind and downwind nodes for each face
        edge_weight_back_(all) = 0.5;
        edge_weight_front_(all) = 0.5;

        // if averaging is the required method we just return
        if( spatial_weighting==weightAveraging ){
            return;
        }

        // determine fluxes using averaging
        SpatialWeightType tmp = spatial_weighting;
        spatial_weighting = weightAveraging;
        preprocess_evaluation(t, m, sol, deriv);
        spatial_weighting = tmp;

        // find the spatial weights
        process_spatial_weights(m);
    }

    template<>
    double Physics::compute_mass(const mesh::Mesh& m, const TVecDevice &u) {
        double total_mass = 0.;
        /*
        const double* source = reinterpret_cast<const double*>(&u[0]);
        vdPackI(m.nodes(), &source[0], 1, &h_vec[0]);
        process_volumes_psk( m );
        for(int i=0; i<m.local_nodes(); i++)
            total_mass += m.volume(i).vol()*theta_vec[i]*rho_vec[i];
        */
        return total_mass;
    }

    template<>
    double Physics::mass_flux_per_time(const mesh::Mesh& m){
        double flux_per_time = 0.;
        /*
        for( int i=m.interior_cvfaces(); i<m.cvfaces(); i++ )
        {
            const mesh::CVFace& cvf = m.cvface(i);
            int boundary_tag = cvf.boundary();
            const BoundaryCondition& BC = boundary_condition_h( boundary_tag );
            double t=0.;
            if( BC.type()==3 )
                flux_per_time -= BC.value(t) * m.cvface(i).area();
        }
        */

        return flux_per_time*constants().rho_0();
    }

} // end namespace fvmpor
