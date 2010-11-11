#include "fvmpor_DAE.h"
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
        res_tmp = TVecDevice(2*m.local_nodes());
        res_tmp_host = TVec(2*m.local_nodes());

        // Set initial values
        set_initial_conditions(t, m);
        TVec h_host( h_vec );
        for( int i=0; i<m.local_nodes(); i++ ){
            u[i].h = h_host[i];
            udash[i].h = 0.;
            udash[i].M = 0.;
        }

        // set M according to pressure 
        process_volumes_psk( m ); // find theta for each volume
        TVec rho_host(rho_vec);
        TVec theta_host(theta_vec);
        for( int i=0; i<m.local_nodes(); i++ ){
            u[i].M = rho_host[i]*theta_host[i];
        }

        // Compute residual
        compute_residual(temp, true);

        // Set initial derivatives
        for(int i=0; i<m.local_nodes(); i++)
            udash[i].M = res_tmp_host[2*i+1];
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
            assert(u[i].M == u[i].M &&
                   u[i].M !=  std::numeric_limits<double>::infinity() &&
                   u[i].M != -std::numeric_limits<double>::infinity()
            );
        }

        // Copy h from solution vector to h_vec and c_vec
        const double* source = reinterpret_cast<const double*>(&u[0]);
        const double* source_p = reinterpret_cast<const double*>(&udash[0]);
        if( CoordTraits<impl::CoordDeviceInt>::is_device() ){
            TVecDevice tmp_vec(source, source + 2*m.local_nodes());
            h_vec(all) = tmp_vec(1,2,2*m.local_nodes());
            M_vec_(all) = tmp_vec(2,2,2*m.local_nodes());
            TVecDevice tmp_vec2(source_p, source_p + 2*m.local_nodes());
            Mp_vec_(all) = tmp_vec2(2,2,2*m.local_nodes());
        }
        else{
            vdPackI(m.nodes(), &source[0], 2, h_vec.data());
            vdPackI(m.nodes(), &source[1], 2, M_vec_.data());
            vdPackI(m.nodes(), &source_p[1], 2, Mp_vec_.data());
        }
        // h and gradient at CV faces
        shape_matrix.matvec( h_vec, h_faces );
        shape_gradient_matrixX.matvec( h_vec, grad_h_faces_.x() );
        shape_gradient_matrixY.matvec( h_vec, grad_h_faces_.y() );
        if (dimension == 3)
            shape_gradient_matrixZ.matvec( h_vec, grad_h_faces_.z() );

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
    void Physics::residual_evaluation( double t, const mesh::Mesh& m,
                                       const_iterator sol, const_iterator deriv,
                                       iterator res)
    {
        // collect fluxes to CVs
        cvflux_matrix.matvec(M_flux_faces, res_tmp.data());

        // add the source terms
        //res_tmp += source_vec;

        // subtract the lhs
        res_tmp.at(0,m.local_nodes()-1) -= Mp_vec_;

        // Dirichlet boundary conditions
        res_tmp.at(dirichlet_nodes_)  = h_dirichlet_;
        res_tmp.at(dirichlet_nodes_) -= h_vec.at(dirichlet_nodes_);
        res_tmp.at(m.local_nodes(), lin::end) = mul(rho_vec, theta_vec);
        res_tmp.at(m.local_nodes(), lin::end) -= M_vec_.at(0, m.local_nodes()-1);

        // copy solution into res
        res_tmp_host.at(all) = res_tmp;
        res_tmp_host.at(0,2,lin::end).dump(reinterpret_cast<double*>(&res[0]));
        res_tmp_host.at(1,2,lin::end).dump(reinterpret_cast<double*>(&res[0])+1);

        /*
        if( CoordTraits<impl::CoordDeviceInt>::is_device() ){
            // copy data to the host
            cudaMemcpy( res_tmp_host.data(), res_tmp.data(),
                        res_tmp.dim()*sizeof(double), cudaMemcpyDeviceToHost
            );
            //res_tmp_host.at(all) = res_tmp;
            // distribute data in residual
            int N = m.local_nodes();
            double *target = reinterpret_cast<double*>(&res[0]);
            vdUnpackI(N,  res_tmp_host.data(),    target,   2);
            vdUnpackI(N,  res_tmp_host.data()+N,  target+1, 2);
        }else{
            int N = m.local_nodes();
            double *target = reinterpret_cast<double*>(&res[0]);
            vdUnpackI(N,  res_tmp.data(),    target,   2);
            vdUnpackI(N,  res_tmp.data()+N,  target+1, 2);
        }
        */
    }

    template<>
    void Physics::preprocess_timestep(double t, const mesh::Mesh& m, const_iterator sol, const_iterator deriv){
        //--------------------------------
        // determine the spatial weights
        //--------------------------------
        // find upwind and downwind nodes for each face
        edge_weight_back_.at(all) = 0.5;
        edge_weight_front_.at(all) = 0.5;

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

        result.M = u[i].M - rho_vec[i]*theta_vec[i];
        return result;
    }

    template<>
    Physics::value_type Physics::dirichlet(double t, const mesh::Node& n) const
    {
        value_type is_dirichlet = {};

        if( is_dirichlet_h_vec_[n.id()] )
            is_dirichlet.h = true;

        // the algebraic variables are treated the same as dirichlet variables
        is_dirichlet.M = true;

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
            //total_mass += m.volume(i).vol()*u[i].M;
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
