#include "fvmpor_ODE.h"
#include "shape.h"
#ifdef PROBLEM_HENRY
#include "henry.h"
#endif
#ifdef PROBLEM_SALT
#include "salt.h"
#endif

#include <stdio.h>

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
    void Physics::initialise(double& t, const mesh::Mesh& m,
                             iterator u, iterator udash, iterator temp,
                             Callback compute_residual)
    {
        util::Timer timer;
        std::cerr << "The mesh has " << m.nodes() << " nodes, " << m.cvfaces() << ", " << m.interior_cvfaces() << " CV faces, " << m.elements() << " elements, and " << m.edges() << " edges."<< std::endl; 

        // allocate storage
        timer.tic();
        initialise_vectors( m );
        std::cerr << "initialise_vectors() : " << timer.toc() << std::endl;

        // allocate working space for the residual computation
        res_tmp = TVecDevice(2*m.local_nodes());
        res_tmp_host = TVec(2*m.local_nodes());

        // Set initial values
        set_initial_conditions(t, m);
        TVec h_vec_host(h_vec_);
        TVec c_vec_host(c_vec_);
        for( int i=0; i<m.local_nodes(); i++ ){
            // we still set h specifically on the assumption that set_initial_conditions() has
            // had a stab at ballpark initial values, and to catch any dirichlet values
            u[i].c = c_vec_host[i];
            u[i].h = h_vec_host[i];
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
        const double* source   = reinterpret_cast<const double*>(&u[0]);
        const double* source_p = reinterpret_cast<const double*>(&udash[0]);
        if( CoordTraits<impl::CoordDeviceInt>::is_device() ){
            TVecDevice tmp(2*m.nodes());
            TVecDevice tmp_p(2*m.nodes());
            cudaMemcpy(tmp.data(),  source,  2*sizeof(double)*m.nodes(), cudaMemcpyHostToDevice);
            cudaMemcpy(tmp_p.data(), source_p, 2*sizeof(double)*m.nodes(), cudaMemcpyHostToDevice);
            c_vec_.at(all) = tmp.at(0,2,lin::end);
            h_vec_.at(all) = tmp.at(1,2,lin::end);
            cp_vec_.at(all) = tmp_p.at(0,2,lin::end);
        }
        else{
            vdPackI(m.nodes(), &source[0],   2, c_vec_.data());
            vdPackI(m.nodes(), &source[1],   2, h_vec_.data());
            vdPackI(m.nodes(), &source_p[0], 2, cp_vec_.data());
        }

        // Compute shape function values and gradients
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
        edge_weight_back_(all) = 0.5;
        edge_weight_front_(all) = 0.5;

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
    void Physics::residual_evaluation( double t, const mesh::Mesh& m,
                                       const_iterator sol, const_iterator deriv,
                                       iterator res)
    {
        int N = m.local_nodes();

        // collect fluxes to CVs
        double factor = constants().rho_0() * constants().eta();
        res_tmp.zero();
        cvflux_matrix.matvec(M_flux_faces_, res_tmp.data()+m.local_nodes());
        cvflux_matrix.matvec(C_flux_faces_, res_tmp.data());

        //res_tmp.at(N,lin::end) -= factor*res_tmp.at(0,N-1);

        res_tmp.at(0,N-1) /= phi_vec_;
        res_tmp.at(0,N-1) -= cp_vec_.at(0,N-1);

        // Dirichlet boundary conditions
        res_tmp.at(N,lin::end).at(dirichlet_h_nodes_)  = dirichlet_h_ - h_vec_.at(dirichlet_h_nodes_);
        res_tmp.at(dirichlet_c_nodes_)  = dirichlet_c_ - c_vec_.at(dirichlet_c_nodes_);

        if( CoordTraits<impl::CoordDeviceInt>::is_device() ){
            cudaMemcpy( res_tmp_host.data(), res_tmp.data(),
                        res_tmp.dim()*sizeof(double), cudaMemcpyDeviceToHost
            );
            // copy from device to host
            //res_tmp_host = res_tmp;
            for(int i=0; i<N; i++){
                res[i].c = res_tmp_host[i];
                res[i].h = res_tmp_host[N+i];
            }
        }else{
            // LIN_DEBUG
            for(int i=0; i<N; i++){
                res[i].c = res_tmp[i];
                res[i].h = res_tmp[N+i];
            }
        }
    }

    template<>
    double Physics::compute_mass(const mesh::Mesh& m, const_iterator u) {
        double total_mass = 0.;
        /*
        const double* source = reinterpret_cast<const double*>(&u[0]);
        vdPackI(m.nodes(), &source[0], 1, &h_vec_[0]);
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
