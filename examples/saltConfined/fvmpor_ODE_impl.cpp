#include "fvmpor_ODE.h"
#include "shape.h"
#ifdef PROBLEM_HENRY
#include "henry.h"
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
        std::cerr << "The mesh has " << m.nodes() << " nodes, " << m.cvfaces() << ", " << m.interior_cvfaces() << " CV faces, " << m.elements() << " elements, and " << m.edges() << " edges."<< std::endl; 

        // allocate storage
        initialise_vectors( m );

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

        // rely on the user calling
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
        vdPackI(m.nodes(), &source[0],   2, &c_vec_[0]);
        vdPackI(m.nodes(), &source[1],   2, &h_vec_[0]);
        vdPackI(m.nodes(), &source_p[0], 2, &cp_vec_[0]);

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
        edge_weight_back_.at(all) = 0.5;
        edge_weight_front_.at(all) = 0.5;

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
        res_tmp.zero();

        double factor = constants().rho_0() * constants().eta();

        cvflux_matrix.matvec(M_flux_faces_, res_tmp.data()+m.local_nodes());
        cvflux_matrix.matvec(C_flux_faces_, res_tmp.data());

        ///////////////////////////////
        /*
        std::cerr << std::endl;
        for(int i=0; i<M_flux_faces_.dim(); i++)
            //if(fabs(M_flux_faces_[i])>1e-16)
            //if(  (fabs(res_tmp[m.cvface(i).back().id()+N])>1e-16
            //        || (i<m.interior_cvfaces() && fabs(res_tmp[m.cvface(i).front().id()+N])>1e-16))
            //        && fabs(M_flux_faces_[i])>1e-10
            //  )
            if(m.cvface(i).back().id()==5 || (i<m.interior_cvfaces() && m.cvface(i).front().id()==5))
            {
                fprintf(stderr, "%d (%7g,%7g)\t:\t%15.14g\t%15.14g\n", i,
                            m.cvface(i).centroid().x, m.cvface(i).centroid().y,
                            M_flux_faces_[i], C_flux_faces_[i] );
                fprintf(stderr, "%d (%7g,%7g)\t:\t%15.14g\t%15.14g\n", i,
                            m.cvface(i).centroid().x, m.cvface(i).centroid().y,
                            qdotn_faces_[i], qcdotn_faces_[i] );
            }
        std::cerr << std::endl;
        for(int i=0; i<N; i++){
            //if(m.node(i).point().x==0 || m.node(i).point().x==200)
            if( fabs(res_tmp[i+N]) > 1e-10 )
                fprintf(stderr, "%d (%7g,%7g)\t:\t%5.4g\t%5.4g\n", i, m.node(i).point().x, m.node(i).point().y, res_tmp[i+N], res_tmp[i] );
        }
        */
        ///////////////////////////////
        res_tmp.at(0,N-1) *= factor;

        for(int i=0; i<N; i++)
            res_tmp[i+N] -= res_tmp[i];

        cvflux_matrix.matvec(C_flux_faces_, res_tmp.data());
        res_tmp.at(0,N-1) /= phi_vec_;
        //res_tmp.at(0,N-1) -= cp_vec_.at(0,N-1);
        for(int i=0; i<N; i++)
            res_tmp[i] -= cp_vec_[i];

        // Dirichlet boundary conditions
        for(int i=0; i<dirichlet_h_nodes_.dim(); i++){
            res_tmp.at(dirichlet_h_nodes_[i]+N)  = dirichlet_h_[i] - h_vec_[dirichlet_h_nodes_[i]];
        }
        for(int i=0; i<dirichlet_c_nodes_.dim(); i++){
            res_tmp.at(dirichlet_c_nodes_[i])  = dirichlet_c_[i] - c_vec_[dirichlet_c_nodes_[i]];
        }

        for(int i=0; i<N; i++){
            res[i].c = res_tmp[i];
            res[i].h = res_tmp[N+i];
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        std::cerr << "writing residual to res.m" << std::endl;
        std::ofstream fid("res.m");
        fid.precision(20);
        fid << "resfh = [";
        for(int i=0; i<N; i++)
            fid << res[i].h << " ";
        fid << "]';" << std::endl;
        fid << "resfc = [";
        for(int i=0; i<N; i++)
            fid << res[i].c << " ";
        fid << "]';" << std::endl;
        /////////////////////////////////////////////////////////////////////////////////////////////
    }

    /*
    template<>
    void Physics::residual_evaluation_old( double t, const mesh::Mesh& m,
                                       const_iterator sol, const_iterator deriv,
                                       iterator res)
    {
        int N = m.local_nodes();

        // collect fluxes to CVs
        res_tmp.zero();

        double factor = 1./( constants().rho_0() * (1.+constants().eta()) );

        cvflux_matrix.matvec(M_flux_faces_, res_tmp.data());
        res_tmp.at(0,N-1) *= factor;
        cvflux_matrix.matvec(C_flux_faces_, res_tmp.data()+m.local_nodes());

        ///////////////////////////////
        std::cerr << "residual on device" << std::endl << "============================================" << std::endl;
        for(int i=0; i<N; i++)
            std::cerr <<  i << " : " << res_tmp[i] << " , " << res_tmp[i+N] << std::endl;       
        std::cerr << std::endl;
        ///////////////////////////////

        //res_tmp.at(N,lin::end) -= res_tmp.at(0,N-1);
        for(int i=0; i<N; i++)
            res_tmp[i+N] -= res_tmp[i];

        ///////////////////////////////
        std::cerr << "residual on device" << std::endl << "============================================" << std::endl;
        for(int i=0; i<N; i++)
            std::cerr <<  i << " : " << res_tmp[i] << " , " << res_tmp[i+N] << std::endl;       
        std::cerr << std::endl;
        ///////////////////////////////

        res_tmp.at(0,N-1) /= phi_vec_;

        // subtract the lhs
        res_tmp.at(0,N-1) -= cp_vec_.at(0,N-1);

        // Dirichlet boundary conditions
        for(int i=0; i<dirichlet_h_nodes_.dim(); i++){
            res_tmp.at(dirichlet_h_nodes_[i]+N)  = dirichlet_h_[i] - h_vec_[dirichlet_h_nodes_[i]];
        }
        for(int i=0; i<dirichlet_c_nodes_.dim(); i++){
            res_tmp.at(dirichlet_c_nodes_[i])  = dirichlet_c_[i] - c_vec_[dirichlet_c_nodes_[i]];
        }
        std::cerr << "residual on device" << std::endl << "============================================" << std::endl;
        for(int i=0; i<N; i++)
            std::cerr <<  i << " : " << res_tmp[i] << " , " << res_tmp[i+N] << std::endl;       
        std::cerr << std::endl;
        // copy solution into res
        for(int i=0; i<N; i++){
            res[i].c = res_tmp[i];
            res[i].h = res_tmp[N+i];
        }

        std::cerr << "RESIDUAL" << std::endl << "================================" << std::endl;
        for(int i=0; i<N; i++)
            std::cerr << i << "\t" << sol[i].c << "\t" << sol[i].h << "\t" << res[i].c << "\t" << res[i].h << std::endl;
        std::cerr << std::endl;
    }
    */

    /*
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
    */

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
