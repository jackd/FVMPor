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

    using mesh::Point;
    using util::DoubleVector;


    template<>
    void Physics::initialise(double& t, const mesh::Mesh& m,
                             iterator u, iterator udash, iterator temp,
                             Callback compute_residual)
    {
        // allocate storage
        initialise_vectors( m );

        // allocate working space for the residual computation
        res_tmp = TVecDevice(m.local_nodes());
        res_tmp_host = TVec(m.local_nodes());

        // Set initial values
        set_initial_conditions(t, m);
        TVec h_vec_host(h_vec);
        for( int i=0; i<m.local_nodes(); i++ ){
            u[i].h = h_vec_host[i];
        }

        // Compute residual
        compute_residual(temp, true);

        // Set initial derivatives
        TVec ahh_vec_host(ahh_vec);
        for (int i = 0; i < m.local_nodes(); ++i) {
            const mesh::Node& n = m.node(i);
            Point p = n.point();
            double x = p.x;

            if( !is_dirichlet_h_vec_[i] ){
                if( ahh_vec[i] )
                    udash[i].h = temp[i].h/ahh_vec_host[i];
            }
        }
    }

    template<>
    void Physics::preprocess_evaluation(double t, const mesh::Mesh& m,
                                        const_iterator u, const_iterator udash)
    {
        ++num_calls;
        util::Timer timer;

        for (int i = 0; i < m.nodes(); ++i) {
            assert(    u[i].h ==  u[i].h
                    && u[i].h !=  std::numeric_limits<double>::infinity()
                    && u[i].h != -std::numeric_limits<double>::infinity()
            );
        }

        timer.tic();
        // Copy h and c over to h_vec and c_vec
        const double* source   = reinterpret_cast<const double*>(&u[0]);
        const double* sourcep  = reinterpret_cast<const double*>(&udash[0]);
        double* target  = h_vec.data();
        double* targetp = hp_vec_.data();
        // this is a dirty hack that will be replaced with a clean minlin library call
        if( CoordTraits<impl::CoordDeviceInt>::is_device() ){
            cudaMemcpy(target,  source,  sizeof(double)*m.nodes(), cudaMemcpyHostToDevice);
            cudaMemcpy(targetp, sourcep, sizeof(double)*m.nodes(), cudaMemcpyHostToDevice);
        }
        else{
            vdPackI(m.nodes(), &source[0],  1, target);
            vdPackI(m.nodes(), &sourcep[0], 1, targetp);
        }

        // Compute shape function values and gradients
        shape_matrix.matvec( h_vec, h_faces );
        shape_gradient_matrixX.matvec( h_vec, grad_h_faces_.x() );
        shape_gradient_matrixY.matvec( h_vec, grad_h_faces_.y() );
        if (dimension == 3){
            shape_gradient_matrixZ.matvec( h_vec, grad_h_faces_.z() );
        }

        // determine the p-s-k values
        //timer.tic();
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
                                       const_iterator sol, const_iterator deriv,
                                       iterator res)
    {
        // collect fluxes to CVs
        TVecDevice res_tmp(m.local_nodes());
        cvflux_matrix.matvec(M_flux_faces, res_tmp);

        // add the source terms here
        //res_tmp += source_vec;

        // subtract the lhs
        //res_tmp -= mul(hp_vec_.at(0,res_tmp.dim()-1),ahh_vec);
        res_tmp -= mul(hp_vec_,ahh_vec);

        // Dirichlet boundary conditions
        res_tmp.at(dirichlet_nodes_)  = h_dirichlet_;
        res_tmp.at(dirichlet_nodes_) -= h_vec.at(dirichlet_nodes_);

        // copy solution into res
        if( CoordTraits<impl::CoordDeviceInt>::is_device() ){
            cudaMemcpy( reinterpret_cast<double*>(&res[0]), res_tmp.data(),
                        res_tmp.dim()*sizeof(double), cudaMemcpyDeviceToHost
            );
        }else{
            for(int i=0; i<m.local_nodes(); i++)
                res[i].h = res_tmp[i];
        }
    }

    template<>
    void Physics::preprocess_timestep( double t, const mesh::Mesh& m, const_iterator sol,
                                       const_iterator deriv)
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
    double Physics::compute_mass(const mesh::Mesh& m, const_iterator u) {
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
