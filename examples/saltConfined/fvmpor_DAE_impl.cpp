#include "fvmpor_DAE.h"
#ifdef PROBLEM_HENRY
#include "henry.h"
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

    template<>
    Physics::value_type Physics::flux(double t, const mesh::CVFace& cvf, const_iterator sol) const{
        Physics::value_type result;
        int id = cvf.id();

        result.h = M_flux_faces_[id];
        result.c = C_flux_faces_[id];
        result.M = std::numeric_limits<double>::quiet_NaN();
        result.C = std::numeric_limits<double>::quiet_NaN();

        return result;
    }

    template<>
    Physics::value_type Physics::boundary_flux(double t, const mesh::CVFace& cvf, const_iterator sol) const{
        Physics::value_type result;
        int id = cvf.id();

        result.h = M_flux_faces_[id];
        result.c = C_flux_faces_[id];
        result.M = std::numeric_limits<double>::quiet_NaN();
        result.M = std::numeric_limits<double>::quiet_NaN();

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
            u[i].h = h_vec_[i];
            u[i].c = c_vec_[i];
            udash[i].M = 0;
            udash[i].C = 0;
        }

        // set M and C according to pressure and concentration 
        process_volumes_psk( m ); // find theta for each volume
        for( int i=0; i<m.local_nodes(); i++ ){
            u[i].M = rho_vec_[i]*phi_vec_[i];
            u[i].C = c_vec_[i]*phi_vec_[i];
        }

        // Compute residual
        compute_residual(temp, true);

        // determine the derivative coefficients
        process_derivative_coefficients( m );

        // Set initial derivatives
        for( int f=0; f<m.interior_cvfaces(); f++ ){
            int front_id = m.cvface(f).front().id();
            double vol = m.volume(front_id).vol();
            if (front_id < m.local_nodes()){
                udash[front_id].M += M_flux_faces_[f]/vol;
                udash[front_id].C += C_flux_faces_[f]/vol;
            }
            int back_id = m.cvface(f).back().id();
            vol = m.volume(back_id).vol();
            if (back_id < m.local_nodes()){
                udash[back_id].M -= M_flux_faces_[f]/vol;
                udash[back_id].C -= C_flux_faces_[f]/vol;
            }
        }
        for( int f=m.interior_cvfaces(); f<m.cvfaces(); f++){
            int back_id = m.cvface(f).back().id();
            double vol = m.volume(back_id).vol();
            if (back_id < m.local_nodes()){
                udash[back_id].M -= M_flux_faces_[f]/vol;
                udash[back_id].C -= C_flux_faces_[f]/vol;
            }
        }
        /*
        for( int i=0; i<m.local_nodes(); i++ ){
            if( m.node(i).boundary() )
                std::cout << i << " " << m.node(i).boundary() << " " << m.node(i).point() << "\t" << u[i].h << "\t" << u[i].c << "\t" << u[i].M << "\t" << u[i].C << "\t" << udash[i].h << "\t" << udash[i].c << "\t" << udash[i].M << "\t" << udash[i].C << std::endl;
        }
        */
    }

    template<>
    void Physics::preprocess_evaluation(double t, const mesh::Mesh& m,
                                        const_iterator u, const_iterator udash)
    {
        ++num_calls;

        for (int i = 0; i < m.nodes(); ++i) {
            assert(u[i].h == u[i].h &&
                   u[i].h !=  std::numeric_limits<double>::infinity() &&
                   u[i].h != -std::numeric_limits<double>::infinity()
            );
            assert(u[i].c == u[i].c &&
                   u[i].c !=  std::numeric_limits<double>::infinity() &&
                   u[i].c != -std::numeric_limits<double>::infinity()
            );
            assert(u[i].M == u[i].M &&
                   u[i].M !=  std::numeric_limits<double>::infinity() &&
                   u[i].M != -std::numeric_limits<double>::infinity()
            );
            assert(u[i].C == u[i].C &&
                   u[i].C !=  std::numeric_limits<double>::infinity() &&
                   u[i].C != -std::numeric_limits<double>::infinity()
            );
        }

        // Copy h from solution vector to h_vec and c_vec
        const double* source = reinterpret_cast<const double*>(&u[0]);
        vdPackI(m.nodes(), &source[0], 4, &h_vec_[0]);
        vdPackI(m.nodes(), &source[1], 4, &c_vec_[0]);

        // h and gradient at CV faces
        shape_matrix.matvec( h_vec_, h_faces_ );
        shape_matrix.matvec( c_vec_, c_faces_ );
        shape_gradient_matrixX.matvec( h_vec_, grad_h_faces_.x() );
        shape_gradient_matrixY.matvec( h_vec_, grad_h_faces_.y() );
        if (dimension == 3)
            shape_gradient_matrixZ.matvec( h_vec_, grad_h_faces_.z() );
        shape_gradient_matrixX.matvec( c_vec_, grad_c_faces_.x() );
        shape_gradient_matrixY.matvec( c_vec_, grad_c_faces_.y() );
        if (dimension == 3)
            shape_gradient_matrixZ.matvec( c_vec_, grad_c_faces_.z() );

        // determine the p-s-k values
        process_volumes_psk( m );
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
        edge_weight_back_ = 0.5;
        edge_weight_front_ = 0.5;

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
    Physics::value_type Physics::lhs(double t, const mesh::Volume& volume,
                                     const_iterator u, const_iterator udash) const
    {
        int i = volume.id();
        value_type result;

        result.h = udash[i].M;

        // Dirichlet conditions
        if( is_dirichlet_h_vec_[i] ){
            const BoundaryCondition& bc = boundary_condition_h(is_dirichlet_h_vec_[i]);
            if( bc.type()==1 || bc.type()==7 ){
                result.h = u[i].h - bc.value(t);
            }
            else{
                double el = dimension == 2 ? volume.node().point().y : volume.node().point().z;
                result.h = u[i].h - bc.hydrostatic(t, el);
            }
        }
        if( is_dirichlet_c_vec_[i] ){
            const BoundaryCondition& bc = boundary_condition_c(is_dirichlet_c_vec_[i]);
            assert( bc.type()==1);
            result.c = u[i].c - bc.value(t);
        }

        result.M = u[i].M - rho_vec_[i]*phi_vec_[i];
        result.C = u[i].C - c_vec_[i]*phi_vec_[i];
        return result;
    }

    template<>
    Physics::value_type Physics::dirichlet(double t, const mesh::Node& n) const
    {
        value_type is_dirichlet = {};

        if( is_dirichlet_h_vec_[n.id()] )
            is_dirichlet.h = true;
        if( is_dirichlet_c_vec_[n.id()] )
            is_dirichlet.c = true;

        // the algebraic variables are treated the same as dirichlet variables
        is_dirichlet.M = true;
        is_dirichlet.C = true;

        return is_dirichlet;
    }

    template<>
    double Physics::compute_mass(const mesh::Mesh& m, const_iterator u) {
        double total_mass = 0.;
        /*
        const double* source = reinterpret_cast<const double*>(&u[0]);
        vdPackI(m.nodes(), &source[0], 2, &h_vec[0]);
        process_volumes_psk( m );
        for(int i=0; i<m.local_nodes(); i++)
            total_mass += m.volume(i).vol()*u[i].M;
            //total_mass += m.volume(i).vol()*theta_vec[i]*rho_vec_[i];
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
