#include "fvmpor_ODE.h"
#include "shape.h"
#ifdef PROBLEM_CASSION
#include "cassion.h"
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

    using mesh::Point;
    using util::DoubleVector;

    template<>
    Physics::value_type Physics::flux(double t, const mesh::CVFace& cvf, const_iterator sol) const{
        Physics::value_type result;
        int id = cvf.id();

        result.h = M_flux_faces[id];
        result.c = C_flux_faces[id];

        return result;
    }

    template<>
    Physics::value_type Physics::boundary_flux(double t, const mesh::CVFace& cvf, const_iterator sol) const{
        Physics::value_type result;
        int id = cvf.id();

        result.h = M_flux_faces[id];
        result.c = C_flux_faces[id];

        return result;
    }

    double det( double a11, double a12, double a21, double a22 )
    {
        return a11*a22 - a21*a12;
    }

    template<>
    void Physics::initialise(double& t, const mesh::Mesh& m,
                             iterator u, iterator udash, iterator temp,
                             Callback compute_residual)
    {
        // allocate storage
        initialise_vectors( m );

        // Set initial values
        set_initial_conditions(t, m);
        for( int i=0; i<m.local_nodes(); i++ ){
            u[i].h = h_vec[i];
            u[i].c = c_vec[i];
        }

        // Compute residual
        compute_residual(temp, true);

        // Set initial derivatives
        for (int i = 0; i < m.local_nodes(); ++i) {
            const mesh::Node& n = m.node(i);
            Point p = n.point();
            double x = p.x;
        }
    }

    template<>
    void Physics::preprocess_evaluation(double t, const mesh::Mesh& m,
                                        const_iterator u, const_iterator udash)
    {
        ++num_calls;

        for (int i = 0; i < m.nodes(); ++i) {
            assert( u[i].h == u[i].h && u[i].h !=  std::numeric_limits<double>::infinity() && u[i].h != -std::numeric_limits<double>::infinity() );
            assert( u[i].c == u[i].c && u[i].c !=  std::numeric_limits<double>::infinity() && u[i].c != -std::numeric_limits<double>::infinity() );
        }

        // Copy h and c over to h_vec and c_vec
        const double* source = reinterpret_cast<const double*>(&u[0]);
        vdPackI(m.nodes(), &source[0], 2, &h_vec[0]);
        vdPackI(m.nodes(), &source[1], 2, &c_vec[0]);

        // Compute shape function values and gradients
        shape_matrix.matvec( h_vec, h_faces );
        shape_matrix.matvec( c_vec, c_faces );
        shape_gradient_matrixX.matvec( h_vec, grad_h_faces_.x() );
        shape_gradient_matrixY.matvec( h_vec, grad_h_faces_.y() );
        if (dimension == 3)
            shape_gradient_matrixZ.matvec( h_vec, grad_h_faces_.z() );
        shape_gradient_matrixX.matvec( c_vec, grad_c_faces_.x() );
        shape_gradient_matrixY.matvec( c_vec, grad_c_faces_.y() );
        if (dimension == 3)
            shape_gradient_matrixZ.matvec( c_vec, grad_c_faces_.z() );

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
    void Physics::preprocess_timestep(double t, const mesh::Mesh& m, const_iterator sol, const_iterator deriv){
        //--------------------------------
        // determine the spatial weights
        //--------------------------------
        // find upwind and downwind nodes for each face
        edge_weight_back_ = 0.5;
        edge_weight_front_ = 0.5;

        // if averaging is the required method we just return
        if( spatial_weighting==weightAveraging )
            return;

        // determine fluxes using averaging
        SpatialWeightType tmp = spatial_weighting;
        spatial_weighting = weightAveraging;
        preprocess_evaluation(t, m, sol, deriv);
        spatial_weighting = tmp;

        // find the spatial weights
        process_spatial_weights(m);
    }

    template<>
    Physics::value_type Physics::lhs(double t, const mesh::Volume& volume,
                                     const_iterator u, const_iterator udash) const
    {
        int i = volume.id();
        value_type result;

        //result.h = ahh_vec[i]*udash[i].h;
        result.h = ahh_*udash[i].h + ahc_*udash[i].c;
        result.c = ach_*udash[i].h + acc_*udash[i].c;

        // Dirichlet conditions
        if( is_dirichlet_h_vec[i] ){
            const BoundaryCondition& bc = boundary_condition_h(is_dirichlet_h_vec[i]);
            if( bc.type()==1 ){
                result.h = u[i].h - bc.value(t);
            }
            else{
                double el = dimension == 2 ? volume.node().point().y : volume.node().point().z;
                if(bc.type()==4)
                    result.h = u[i].h - bc.hydrostatic(t, el);
                else{
                    result.h = u[i].h - bc.hydrostatic_shore(t, el);
                }
            }
        }
        if( is_dirichlet_c_vec[i] ){
            const BoundaryCondition& bc = boundary_condition_c(is_dirichlet_c_vec[i]);
            assert(bc.type()==1);
            result.c = u[i].c - bc.value(t);
        }

        return result;
    }

    template<>
    Physics::value_type Physics::dirichlet(double t, const mesh::Node& n) const
    {
        value_type is_dirichlet = {};

        if( is_dirichlet_h_vec[n.id()] )
            is_dirichlet.h = true;
        if( is_dirichlet_c_vec[n.id()] )
            is_dirichlet.c = true;
        return is_dirichlet;
    }

    template<>
    double Physics::compute_mass(const mesh::Mesh& m, const_iterator u) {
        double total_mass = 0.;
        const double* source = reinterpret_cast<const double*>(&u[0]);
        vdPackI(m.nodes(), &source[0], 1, &h_vec[0]);
        process_volumes_psk( m );
        for(int i=0; i<m.local_nodes(); i++)
            total_mass += m.volume(i).vol()*theta_vec[i]*rho_vec[i];
        return total_mass;
    }

    template<>
    double Physics::mass_flux_per_time(const mesh::Mesh& m){
        double flux_per_time = 0.;
        for( int i=m.interior_cvfaces(); i<m.cvfaces(); i++ )
        {
            const mesh::CVFace& cvf = m.cvface(i);
            int boundary_tag = cvf.boundary();
            const BoundaryCondition& BC = boundary_condition_h( boundary_tag );
            double t=0.;
            if( BC.type()==3 )
                flux_per_time -= BC.value(t) * m.cvface(i).area();
        }

        return flux_per_time*constants().rho_0();
    }

} // end namespace fvmpor
