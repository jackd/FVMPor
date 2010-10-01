#include "fvmpor_ODE.h"
#include "shape.h"
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
    Physics::value_type Physics::flux(double t, const mesh::CVFace& cvf, const_iterator sol) const{
        Physics::value_type result;
        int id = cvf.id();

        //result.h = M_flux_faces[id];
        result.h = M_flux_faces_host[id];

        return result;
    }

    template<>
    Physics::value_type Physics::boundary_flux(double t, const mesh::CVFace& cvf, const_iterator sol) const{
        Physics::value_type result;
        int id = cvf.id();

        //result.h = M_flux_faces[id];
        result.h = M_flux_faces_host[id];

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
        util::Timer timer;
        timer.tic();

        // allocate storage
        initialise_vectors( m );

        // Set initial values
        set_initial_conditions(t, m);
        for( int i=0; i<m.local_nodes(); i++ ){
            u[i].h = h_vec[i];
        }

        // Compute residual
        compute_residual(temp, true);

        // Set initial derivatives
        for (int i = 0; i < m.local_nodes(); ++i) {
            const mesh::Node& n = m.node(i);
            Point p = n.point();
            double x = p.x;

            if( !is_dirichlet_h_vec_[i] ){
                if( ahh_vec[i] )
                    udash[i].h = temp[i].h/ahh_vec_host[i];
            }
        }
        //std::cout << "INITIALISATION TOOK " << timer.toc() << std::endl;
    }

    template<>
    void Physics::preprocess_evaluation(double t, const mesh::Mesh& m,
                                        const_iterator u, const_iterator udash)
    {
        util::Timer timer, gtimer;

        ++num_calls;

        gtimer.tic();
        for (int i = 0; i < m.nodes(); ++i) {
            assert( u[i].h == u[i].h && u[i].h !=  std::numeric_limits<double>::infinity() && u[i].h != -std::numeric_limits<double>::infinity() );
        }

        // Copy h and c over to h_vec and c_vec
        timer.tic();
        const double* source = reinterpret_cast<const double*>(&u[0]);
        double* target = h_vec.data();
        // this is a dirty hack that will be replaced with a clean minlin library call
        if( CoordTraits<impl::CoordDeviceInt>::is_device() ){
            cudaMemcpy(target, source, sizeof(double)*m.nodes(), cudaMemcpyHostToDevice);
        }
        else
            vdPackI(m.nodes(), &source[0], 1, target);
        double t1=timer.toc();

        // Compute shape function values and gradients
        timer.tic();
        shape_matrix.matvec( h_vec, h_faces );
        shape_gradient_matrixX.matvec( h_vec, grad_h_faces_.x() );
        shape_gradient_matrixY.matvec( h_vec, grad_h_faces_.y() );
        if (dimension == 3){
            shape_gradient_matrixZ.matvec( h_vec, grad_h_faces_.z() );
        }
        double t2 = timer.toc();

        // determine the p-s-k values
        //timer.tic();
        timer.tic();
        process_volumes_psk( m );
        double t3 = timer.toc();

        // determine derivative coefficients
        timer.tic();
        process_derivative_coefficients( m );
        double t4 = timer.toc();

        //std::cout << t1 << " " << t2 << " " << t3 << " " << t4 << std::endl;

        // density at faces using shape functions
        timer.tic();
        process_faces_shape( m );
        double t5 = timer.toc();

        // Compute psk values at faces using upwinding/limiting
        timer.tic();
        process_faces_lim( m );
        double t6 = timer.toc();

        // compute fluxes
        timer.tic();
        process_fluxes( t, m );
        double t7 = timer.toc();

        timer.tic();
        if( CoordTraits<impl::CoordDeviceInt>::is_device() ){
            cudaMemcpy(M_flux_faces_host.data(), M_flux_faces.data(), M_flux_faces.dim()*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(ahh_vec_host.data(), ahh_vec.data(), ahh_vec.dim()*sizeof(double), cudaMemcpyDeviceToHost);
        }
        else{
            M_flux_faces_host(all) = M_flux_faces;
            ahh_vec_host(all) = ahh_vec;
        }
        double t8 = timer.toc();

        //std::cout << "PREPROCESS EVALUATION TOOK " << gtimer.toc() << std::endl;
        //std::cerr << "PREPROCESS : " << t1 << ", " <<  t2 << ", " <<  t3 << ", " <<  t4 << ", " <<  t5 << ", " <<  t6 << ", " <<  t7 << " " << t8 << " = " << t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 << std::endl; 
        /*
        std::ofstream fid;
        if( CoordTraits<impl::CoordDeviceInt>::is_device() )
            fid.open("preG.txt");
        else
            fid.open("preC.txt");
        PRINT(fid,ahh_vec)
        PRINT(fid,M_flux_faces)
        exit(0);
        */
    }

    template<>
    void Physics::preprocess_timestep(double t, const mesh::Mesh& m, const_iterator sol, const_iterator deriv){
        util::Timer timer;
        timer.tic();
        //--------------------------------
        // determine the spatial weights
        //--------------------------------
        // find upwind and downwind nodes for each face
        edge_weight_back_(all) = 0.5;
        edge_weight_front_(all) = 0.5;

        // if averaging is the required method we just return
        if( spatial_weighting==weightAveraging ){
            //std::cout << "PREPROCESS TIMESTEP TOOK " << timer.toc() << std::endl;
            return;
        }

        // determine fluxes using averaging
        SpatialWeightType tmp = spatial_weighting;
        spatial_weighting = weightAveraging;
        preprocess_evaluation(t, m, sol, deriv);
        spatial_weighting = tmp;

        // find the spatial weights
        process_spatial_weights(m);
        //std::cout << "PREPROCESS TIMESTEP TOOK " << timer.toc() << std::endl;
    }

    template<>
    Physics::value_type Physics::lhs(double t, const mesh::Volume& volume,
                                     const_iterator u, const_iterator udash) const
    {
        int i = volume.id();
        value_type result;

        result.h = ahh_vec_host[i]*udash[i].h;

        // Dirichlet conditions
        if( is_dirichlet_h_vec_[i] ){
            const BoundaryCondition& bc = boundary_condition_h(is_dirichlet_h_vec_[i]);
            if( bc.type()==1 || bc.type()==7 ){
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
        return result;
    }

    template<>
    Physics::value_type Physics::dirichlet(double t, const mesh::Node& n) const
    {
        value_type is_dirichlet = {};

        if( is_dirichlet_h_vec_[n.id()] )
            is_dirichlet.h = true;
        return is_dirichlet;
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
