#include "fvmpor.h"
#include "shape.h"

#include <mkl_vml_functions.h>

#include <util/doublevector.h>
#include <fvm/mesh.h>

#include <algorithm>
#include <vector>
#include <limits>
#include <utility>

#include <math.h>

// For debugging
#include <iostream>
#include <ostream>
#include <iterator>

namespace fvmpor {

    double relative_permeability(double h, const PhysicalZone& props)
    {
        if( h>=0. )
            return 1.;

        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;

        double Se = 1./ pow( 1. + pow(alphaVG*fabs(h), nVG), mVG );
        return sqrt(Se) * pow( 1. - pow(1.-pow(Se,1./mVG), mVG) , 2. );
    }

    double density(double h, const Constants& constants)
    {
        double beta = constants.beta();
        double rho_0 = constants.rho_0();
        double g = constants.g();

        return rho_0*(1. + rho_0*g*beta*h);
    }

    double porosity(double h, double por_0, const Constants& constants)
    {
        double alpha = constants.beta();
        double g = constants.g();
        double rho_0 = constants.rho_0();

        return 1.-(1.-por_0)*( 1.+rho_0*g*alpha*h );
    }

    void saturation( double h, const PhysicalZone &props, double &Sw, double &dSw, double &theta )
    {
        if( h>=0. )
        {
            Sw  = 1.;
            dSw = 0.;
            theta = props.phi;
            return;
        }
        double Se;
        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;
        double S_r = props.S_r;
        double phi = props.phi;

        Se = 1./ pow( 1. + pow(alphaVG*fabs(h), nVG), mVG );
        Sw = S_r + (1.-S_r)*Se;
        theta = phi*Sw;
        dSw = pow(alphaVG,nVG) * (1.-S_r) * (nVG-1.) * pow(fabs(h),nVG-1.) / pow(1.+pow(alphaVG*fabs(h),nVG), mVG+1.);
    }

    double moisture_content(double h, const PhysicalZone& props)
    {
        if( h>=0. )
            return props.phi;

        double Se;
        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;
        double S_r = props.S_r;
        double phi = props.phi;

        Se = 1. / pow( 1. + pow(alphaVG*fabs(h), nVG), mVG );
        return phi*(S_r + (1.-S_r)*Se);
    }

    /*
    using mesh::Point;
    using util::DoubleVector;

    double density(double h, const Constants& constants)
    {
        double beta = constants.beta();
        double rho_0 = constants.rho_0();
        double g = constants.g();

        return rho_0*(1. + rho_0*g*beta*h);
    }

    template <typename TVec>
    void density(TVec& h, TVec& rho, const Constants& constants)
    {
        double beta = constants.beta();
        double rho_0 = constants.rho_0();
        double g = constants.g();

        if( beta ){
            rho = h;
            rho *= rho_0*rho_0*g*beta;
            rho += rho_0;
        }else{
            rho = rho_0;
        }
    }

    double porosity(double h, double por_0, const Constants& constants)
    {
        double alpha = constants.beta();
        double g = constants.g();
        double rho_0 = constants.rho_0();

        return 1.-(1.-por_0)*( 1.+rho_0*g*alpha*h );
    }

    template <typename TVec>
    void porosity(TVec& h, TVec& phi, TVec& dphi, const PhysicalZone& props, const Constants& constants)
    {
        double g = constants.g();
        double rho_0 = constants.rho_0();
        double phi_0 = props.phi;
        double alpha = props.alpha;

        // porosity
        if(alpha==0.){
            phi = phi_0;
            dphi = 0.;
        }
        else{
            phi = h;
            phi *= (phi_0-1.)*rho_0*g*alpha;
            phi += 1.;
            dphi = (phi_0-1.)*rho_0*g*alpha;
        }
    }

    double relative_permeability(double h, const PhysicalZone& props)
    {
        if( h>=0. )
            return 1.;

        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;

        double Se = 1./ pow( 1. + pow(alphaVG*fabs(h), nVG), mVG );
        return sqrt(Se) * pow( 1. - pow(1.-pow(Se,1./mVG), mVG) , 2. );
    }

    template <typename TVec>
    void saturation( TVec& h, const PhysicalZone &props, TVec &Se, TVec &dSw, TVec &krw )
    {
        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;
        double S_r = props.S_r;
        double phi = props.phi;

        // if a = (alpha*|h|)^n, and b = 1+a

        // set dSw = a
        dSw = h;
        dSw *= -alphaVG;
        dSw.raise_to(nVG);

        // Se = 1/b
        Se = 1.;
        Se += dSw;
        krw = -1.;
        krw /= Se;

        // dSw /= b
        dSw /= Se;
        
        // Se = 1/(b^m)
        // this is the final value for Se
        Se.raise_to(-mVG);

        // find dSw
        dSw *= Se;
        dSw /= h;
        dSw *= -(1-S_r)*(nVG-1);

        // find krw
        krw += 1.;
        krw.raise_to(mVG);
        krw -= 1.;
        krw *= krw;
        krw.by_sqroot(Se);

        // now override values for saturated h
        int n=h.size();
        for(int i=0; i<n; i++){
            if(h[i]>=0.){
                dSw[i] = 0.;
                Se[i] = 1.;
                krw[i] = 1.;
            }
        }
    }

    void saturation( double h, const PhysicalZone &props, double &Sw, double &dSw, double &theta )
    {
        if( h>=0. )
        {
            Sw  = 1.;
            dSw = 0.;
            theta = props.phi;
            return;
        }
        double Se;
        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;
        double S_r = props.S_r;
        double phi = props.phi;

        Se = 1./ pow( 1. + pow(alphaVG*fabs(h), nVG), mVG );
        Sw = S_r + (1.-S_r)*Se;
        theta = phi*Sw;
        dSw = pow(alphaVG,nVG) * (1.-S_r) * (nVG-1.) * pow(fabs(h),nVG-1.) / pow(1.+pow(alphaVG*fabs(h),nVG), mVG+1.);
    }

    double moisture_content(double h, const PhysicalZone& props)
    {
        if( h>=0. )
            return props.phi;

        double Se;
        double alphaVG = props.alphaVG;
        double nVG = props.nVG;
        double mVG = props.mVG;
        double S_r = props.S_r;
        double phi = props.phi;

        Se = 1. / pow( 1. + pow(alphaVG*fabs(h), nVG), mVG );
        return phi*(S_r + (1.-S_r)*Se);
    }

    template <typename TVec>
    void VarSatPhysicsImpl<TVec>::initialise_vectors( const mesh::Mesh &m ){
        dimension = m.dim();

        node_comm_.set_pattern( "NP_double", m.node_pattern() );

        // set physical properties
        set_constants();
        set_physical_zones();
        set_boundary_conditions();

        // initialise space for storing p-s-k values
        int N = m.nodes();
        Sw_vec.resize(N);
        dSw_vec.resize(N);
        rho_vec.resize(N);
        theta_vec.resize(N);
        phi_vec.resize(N);
        dphi_vec.resize(N);

        rho_faces_lim.resize(m.interior_cvfaces());
        krw_faces_lim.resize(m.interior_cvfaces());
        rho_faces.resize(m.interior_cvfaces());

        // spatial weightings
        CV_up.resize(m.local_nodes());
        CV_flux.resize(m.nodes()); 
        CV_flux_comm_tag = node_comm_.vec_add(CV_flux.data());

        edge_up.resize(m.edges());
        //edge_2up.resize(m.edges());
        edge_down.resize(m.edges());
        edge_flux.resize(m.edges());

        M_flux_faces.resize(m.cvfaces());
        qdotn_faces.resize(m.cvfaces());

        // initialise space for derivative coefficients
        int NL = m.local_nodes();
        ahh_vec.resize( NL );

        cvface_tags.resize( m.interior_cvfaces() );
        for( int i=0; i<m.interior_cvfaces(); i++ )
            cvface_tags[i] = m.cvface(i).element().physical_tag();

        // tag dirichlet nodes
        // also tag nodes that lie on seepage faces
        // assumes that if there is more than one seepage face, they all have the same tag
        is_dirichlet_h_vec.resize(m.local_nodes());
        for( int i=0; i<m.local_nodes(); i++ ){
            const mesh::Node& n = m.node(i);
            // look for dirichlet tags attached to the node
            for( int j=0; j<n.boundaries(); j++ ){
                int tag = n.boundary(j);
                if( boundary_condition_h(tag).is_dirichlet() ){
                    is_dirichlet_h_vec[i] = tag;
                }
            }
            // search for seepage tags on the node
            // only applied if the node is not also dirichlet - dirichlet conditions always take precedence
            if( !is_dirichlet_h_vec[i] ){
                for( int j=0; j<n.boundaries(); j++ ){
                    int tag = n.boundary(j);
                    if( boundary_condition_h(tag).type()==7 || boundary_condition_h(tag).type()==8 ){
                        seepage_nodes.push_back(i);
                        is_dirichlet_h_vec[i] = tag;
                        seepage_tag = tag;
                        break;
                    }
                }
            }
        }

        // initialise vectors used in calculating derived quantities such as saturation
        // allocate room for each of the arrays
        std::set<int> zones;
        for(int i=0; i<m.elements(); i++)
            zones.insert(m.element(i).physical_tag());
        int num_zones = zones.size();
        int indx=0;
        for( std::set<int>::iterator it=zones.begin(); it!=zones.end(); it++)
            zones_map_[*it] = indx++;

        head_scv.resize( num_zones );
        weight_scv.resize( num_zones );
        index_scv.resize( num_zones );
        phi_scv.resize( num_zones );
        dphi_scv.resize( num_zones );
        Se_scv.resize( num_zones );
        Sw_scv.resize( num_zones );
        theta_scv.resize( num_zones );
        dSw_scv.resize( num_zones );
        krw_scv.resize( num_zones );

        // temp var
        std::vector<std::map<int,int> > nodes_idx;
        nodes_idx.resize(num_zones);
        // compile index and weight information mapping node information to scv information
        //for(int i=0; i<m.local_nodes(); i++){
        for(int i=0; i<m.nodes(); i++){
            const mesh::Volume& cv = m.volume(i);
            double cv_vol = cv.vol();

            std::vector<double> weights(num_zones);
            std::vector<int> counts(num_zones);
            for(int j=0; j<cv.scvs(); j++){
                int tag = zones_map_[cv.scv(j).element().physical_tag()];
                assert(tag<num_zones);
                weights[tag] += cv.scv(j).vol() / cv_vol;
                counts[tag]++;
            }
            for(int j=0; j<num_zones; j++){
                if(counts[j]){
                    weight_scv[j].push_back(weights[j]);
                    index_scv[j].push_back(i);
                    nodes_idx[j][i] = index_scv[j].size()-1;
                }
            }
        }
        // allocate room for head values mapped onto SCVs
        for(int i=0; i<num_zones; i++){
            head_scv[i].resize( index_scv[i].size() );
            phi_scv[i].resize( index_scv[i].size() );
            dphi_scv[i].resize( index_scv[i].size() );
            Se_scv[i].resize( index_scv[i].size() );
            Sw_scv[i].resize( index_scv[i].size() );
            theta_scv[i].resize( index_scv[i].size() );
            dSw_scv[i].resize( index_scv[i].size() );
            krw_scv[i].resize( index_scv[i].size() );
        }

        // this will hold global (face, edge) pairs of each mapped node value in each zone
        std::vector<std::multimap<int, std::pair<int, int> > > faceEdge_map_front;
        std::vector<std::multimap<int, std::pair<int, int> > > faceEdge_map_back;
        faceEdge_map_front.resize(num_zones);
        faceEdge_map_back.resize(num_zones);
        for( int i=0; i<m.edges(); i++ ){
            const std::vector<int>& edge_cvfaces = m.edge_cvface(i);
            int fid = m.edge(i).front().id();
            int bid = m.edge(i).back().id();
            for(int j=0; j<edge_cvfaces.size(); j++){
                int f = edge_cvfaces[j];
                int z = zones_map_[m.cvface(f).element().physical_tag()];
                int n = nodes_idx[z][fid];
                faceEdge_map_front[z].insert(std::pair<int, std::pair<int, int> >( n, std::pair<int, int>(f, i) ));
                n = nodes_idx[z][bid];
                faceEdge_map_back[z].insert(std::pair<int, std::pair<int, int> >( n, std::pair<int, int>(f, i)));
            }
        }

        // should also reserve memory for the vectors
        n_front_.resize(num_zones);
        p_front_.resize(num_zones);
        q_front_.resize(num_zones);
        n_back_.resize(num_zones);
        p_back_.resize(num_zones);
        q_back_.resize(num_zones);
        typedef std::multimap<int, std::pair<int, int> >::iterator idxTypeIt;
        for(int z=0; z<num_zones; z++){
            int len = head_scv[z].size();
            for(int i=0; i<len; i++){
                std::pair<idxTypeIt, idxTypeIt> rng = faceEdge_map_front[z].equal_range(i);
                for( idxTypeIt it=rng.first; it!=rng.second; ++it ){
                    n_front_[z].push_back(i); // local node id
                    q_front_[z].push_back(it->second.first); // global face index
                    p_front_[z].push_back(it->second.second); // global edge index
                }
                rng = faceEdge_map_back[z].equal_range(i);
                for( idxTypeIt it=rng.first; it!=rng.second; ++it ){
                    n_back_[z].push_back(i); // local node id
                    q_back_[z].push_back(it->second.first); // global face index
                    p_back_[z].push_back(it->second.second); // global edge index
                }
            }
        }

        edge_weight_front_.resize(m.edges());
        edge_weight_back_.resize(m.edges());
        edge_weight_back_ = 0.5;
        edge_weight_front_ = 0.5;
        edge_node_front_.resize(m.edges());
        edge_node_back_.resize(m.edges());
        for( int i=0; i<m.edges(); i++){
            edge_node_front_[i] = m.edge(i).front().id();
            edge_node_back_[i] = m.edge(i).back().id();
        }

        // initialise the shape functions
        initialise_shape_functions(m);

        // initialise flux vecs
        qsat_faces_.set(m.interior_cvfaces(), m.dim());

        norm_faces_.set(m.interior_cvfaces(), m.dim());
        for( int i=0; i<m.interior_cvfaces(); i++ ){
            Point nrm = m.cvface(i).normal();
            norm_faces_.x()[i] = nrm.x;
            norm_faces_.y()[i] = nrm.y;
            if( m.dim()==3 )
                norm_faces_.z()[i] = nrm.z;
        }

        K_faces_.set(m.interior_cvfaces(), m.dim());
        for( int i=0; i<m.interior_cvfaces(); i++ ){
            int tag = m.cvface(i).element().physical_tag();
            K_faces_.x()[i] = -physical_zone(tag).K_xx;
            K_faces_.y()[i] = -physical_zone(tag).K_yy;
            if( m.dim()==3 )
                K_faces_.z()[i] = -physical_zone(tag).K_zz;
        }
    }

    template <typename TVec>
    void VarSatPhysicsImpl<TVec>::process_faces_shape( const mesh::Mesh &m )
    {
        density(h_faces, rho_faces, constants());
    }

    template <typename TVec>
    void VarSatPhysicsImpl<TVec>::process_faces_lim( const mesh::Mesh &m )
    {
        for( int e=0; e<m.edges(); e++ ){
            double rho_edge = rho_vec[edge_node_back_[e]]*edge_weight_back_[e] + rho_vec[edge_node_front_[e]]*edge_weight_front_[e];

            const std::vector<int>& edge_cvfaces = m.edge_cvface(e);
            for(int j=0; j<edge_cvfaces.size(); j++){
                int face = edge_cvfaces[j];
                rho_faces_lim[ face ] = rho_edge;
            }
        }
    }

    template <typename TVec>
    void VarSatPhysicsImpl<TVec>::process_fluxes( double t, const mesh::Mesh &m )
    {
        // compute the vector quantity q at each internal CV face
        qsat_faces_.x() = grad_h_faces_.x();
        qsat_faces_.x() *= K_faces_.x();
        qsat_faces_.y() = grad_h_faces_.y();
        if( m.dim()==2 ){
            qsat_faces_.y() += 1.;
        }else{
            qsat_faces_.z() = grad_h_faces_.z();
            qsat_faces_.z() += 1.;
            qsat_faces_.z() *= K_faces_.z();
        }
        qsat_faces_.y() *= K_faces_.y();
        
        qsat_faces_.dot(norm_faces_, qdotn_faces);


        // find the velocity at each CV face
        // these can be written as one operation when subranges are allowed
        for( int i=0; i<m.interior_cvfaces(); i++ ){
            qdotn_faces[i] *= krw_faces_lim[i];
            M_flux_faces[i] = rho_faces_lim[i] * qdotn_faces[i];
        }

        // loop over boundary faces and find fluid flux where explicitly given by BCs
        for( int i=m.interior_cvfaces(); i<m.cvfaces(); i++)
        {
            const mesh::CVFace& cvf = m.cvface(i);

            int boundary_tag = cvf.boundary();
            const BoundaryCondition& BCh = boundary_condition_h( boundary_tag );

            switch( BCh.type() ){
                // prescribed flux
                case 3:
                    qdotn_faces[i] = BCh.value(t) * cvf.area();
                    break;
                // prescribed directional flux
                case 6:
                    qdotn_faces[i] = BCh.flux( t, cvf.normal() ) * cvf.area();
                    break;
                // seepage
                case 7:
                    qdotn_faces[i] = BCh.value(t) * cvf.area();
                    break;
                // seepage/hydrostatic shoreline
                case 8:
                    qdotn_faces[i] = 0. * cvf.area();
                    break;
                default:
                    break;

            }
        }

        for( int i=m.interior_cvfaces(); i<m.cvfaces(); i++)
        {
            const mesh::CVFace& cvf = m.cvface(i);

            int boundary_tag = cvf.boundary();
            const BoundaryCondition& BCh = boundary_condition_h( boundary_tag );

            //if( BCh.is_dirichlet() || BCh.type()==7 )
            if( is_dirichlet_h_vec[cvf.back().id()] )
            {
                // in the case of a dirichlet boundary condition on pressure head we calculate the
                // flux over the control volume face by asserting conservation of mass
                double total_flux = 0.0;
                double total_area = 0.0;

                const mesh::Volume& v = cvf.back().volume();
                for (int ii = 0; ii < v.scvs(); ++ii) {
                    const mesh::SCV& scv = v.scv(ii);
                    for (int j = 0; j < scv.cvfaces(); ++j) {
                        const mesh::CVFace& subcvf = scv.cvface(j);
                        if( subcvf.id()<m.interior_cvfaces() || (!boundary_condition_h(subcvf.boundary()).is_dirichlet() && boundary_condition_h(subcvf.boundary()).type()!=7 && boundary_condition_h(subcvf.boundary()).type()!=8) ){
                            int sign = subcvf.back().id() == v.id() ? 1 : -1;
                            total_flux += sign * qdotn_faces[subcvf.id()];
                        } else{
                            total_area += subcvf.area();
                        }
                    }
                }
                qdotn_faces[i] = -total_flux / total_area * cvf.area();
            }
        }

        // find mass flux over each boundary
        for( int i=m.interior_cvfaces(); i<m.cvfaces(); i++)
        {
            const mesh::CVFace& cvf = m.cvface(i);

            // choose an appropriate concentration and density if the flow is into the domain
            double rho_face = rho_vec[cvf.back().id()];
            if( qdotn_faces[i]>=0 ){
               rho_face = constants().rho_0();
            }

            // now form the flux over the face
            M_flux_faces[i] = rho_face * qdotn_faces[i];
        }
    }

    template <typename TVec>
    void VarSatPhysicsImpl<TVec>::process_spatial_weights(const mesh::Mesh& m){

        // determine the flux over each edge
        flux_lim_matrix.matvec( qdotn_faces, edge_flux );

        switch( spatial_weighting ){
            ////////////////////////////////////////////////////////
            // the upwinding case is simple
            ////////////////////////////////////////////////////////
            case weightUpwind : 
                for(int i=0; i<m.edges(); i++){
                    if( edge_flux[i]<0. ){
                        edge_weight_back_[i] = 0.;
                        edge_weight_front_[i] = 1.;
                        edge_up[i] = m.edge(i).front().id();
                        edge_down[i] = m.edge(i).back().id();
                    }
                    else{
                        edge_weight_back_[i] = 1.;
                        edge_weight_front_[i] = 0.;
                        edge_up[i] = m.edge(i).back().id();
                        edge_down[i] = m.edge(i).front().id();
                    }
                }
                break;
            ////////////////////////////////////////////////////////
            // the flux limitting case takes a bit more work
            ////////////////////////////////////////////////////////
            case weightVanLeer :
                for(int i=0; i<m.edges(); i++){
                    if( edge_flux[i]>0. ){
                        edge_up[i] = m.edge(i).back().id();
                        edge_down[i] = m.edge(i).front().id();
                    }
                    else{
                        edge_up[i] = m.edge(i).front().id();
                        edge_down[i] = m.edge(i).back().id();
                    }
                }

                // find the up node for each CV
                for(int i=0; i<m.local_nodes(); i++){
                    CV_flux[i] = 0.;
                    CV_up[i] = -1.;
                }
                // set the flux into each boundary node to be that from over the boundary
                for(int i=m.interior_cvfaces(); i<m.cvfaces(); i++){
                    int n=m.cvface(i).back().id();
                    CV_flux[n] -= qdotn_faces[i];
                }

                // now find max flux into each CV
                for(int i=0; i<m.edges(); i++){
                    if( edge_node_front_[i]<m.local_nodes() || edge_node_back_[i]<m.local_nodes() ){
                        int CV = edge_down[i];
                        if( CV<m.local_nodes() ){
                            double fl = fabs(edge_flux[i]);
                            if( fl>CV_flux[CV] ){
                                CV_flux[CV] = fl;
                                CV_up[CV] = edge_up[i];
                            }
                        }
                    }
                }

                // verify that each CV was assigned an upwind point
                for(int i=0; i<m.local_nodes(); i++){
                    if(CV_up[i]==-1){
                        CV_up[i] = i;
                    }
                }

                *node_comm_.mpicomm() << "VarSatPhysicsImpl::process_spatial_weights : communicating 2up fluxes values accross subdomain boundaries" << std::endl;
                node_comm_.send(CV_flux_comm_tag);
                node_comm_.recv(CV_flux_comm_tag);

                // find r and sigma for each edge
                for(int i=0; i<m.edges(); i++){
                    if( edge_node_front_[i]<m.local_nodes() || edge_node_back_[i]<m.local_nodes() ){
                        double qup = fabs(edge_flux[i]);
                        double q2up = CV_flux[edge_up[i]];
                        double r = q2up / qup;
                        double sigma;
                        if( qup==0. )
                            sigma = 1.;
                        else if(r>1.e10)
                            sigma = 2.;
                        else
                            sigma = (r+fabs(r)) / (1.+fabs(r));

                        if( edge_flux[i]>0. ){
                            edge_weight_back_[i] = sigma/2.;
                            edge_weight_front_[i] = 1.-sigma/2.;
                        }
                        else{
                            edge_weight_back_[i] = 1.-sigma/2.;
                            edge_weight_front_[i] = sigma/2.;
                        }
                    }
                }
        }
    }

    template <typename TVec>
    void VarSatPhysicsImpl<TVec>::process_volumes_psk( const mesh::Mesh &m )
    {
        double beta = constants().beta();
        double rho_0 = constants().rho_0();
        double g = constants().g();

        // zero out vectors of CV-averaged derived quantities
        phi_vec = 0.;
        dphi_vec = 0.;
        Sw_vec = 0.;
        dSw_vec = 0.;
        theta_vec = 0.;
        krw_faces_lim = 0.;

        // for each zone calucluate the scv-weighted derived quantities and add them to the appropriated CV-averaged vectors
        double T=0.;
        for( std::map<int, int>::iterator it=zones_map_.begin(); it!=zones_map_.end(); it++){
            int zone = (*it).second;
            int indx = (*it).first;
            int n = index_scv.size();
            const PhysicalZone& props = physical_zone(indx);

            // get head data for this zone type
            head_scv[zone].permute_assign(h_vec, index_scv[zone]);

            // find porosity and scale by weights
            porosity(head_scv[zone], phi_scv[zone], dphi_scv[zone], props, constants());

            // the order of scalings below is deliberate so as to reduce flop counts
            // effective saturation
            saturation( head_scv[zone], props, Se_scv[zone], dSw_scv[zone], krw_scv[zone] );

            // saturation
            double S_r = props.S_r;
            Sw_scv[zone].scal_assign( 1.-S_r, Se_scv[zone] );
            Sw_scv[zone] += S_r;
            // moisture content
            theta_scv[zone] = phi_scv[zone];
            theta_scv[zone] *= Sw_scv[zone];

            // copy into global vector
            phi_vec.permute_add_weighted_inverse(phi_scv[zone], index_scv[zone], weight_scv[zone]);
            dphi_vec.permute_add_weighted_inverse(dphi_scv[zone], index_scv[zone], weight_scv[zone]);
            Sw_vec.permute_add_weighted_inverse(Sw_scv[zone], index_scv[zone], weight_scv[zone]);
            dSw_vec.permute_add_weighted_inverse(dSw_scv[zone], index_scv[zone], weight_scv[zone]);
            theta_vec.permute_add_weighted_inverse(theta_scv[zone], index_scv[zone], weight_scv[zone]);

            krw_faces_lim.permute_add_weighted_pqr(krw_scv[zone], q_front_[zone], n_front_[zone], p_front_[zone], edge_weight_front_);
            krw_faces_lim.permute_add_weighted_pqr(krw_scv[zone], q_back_[zone],  n_back_[zone],  p_back_[zone],  edge_weight_back_);
        }
        // find the CV-averaged density - this is much simpler because density is not dependant on material properties
        // of the porous medium
        density(h_vec, rho_vec, constants());
    }

    template <typename TVec>
    void VarSatPhysicsImpl<TVec>::process_derivative_coefficients( const mesh::Mesh &m )
    {
        double rho_0 = constants().rho_0();
        double g = constants().g();
        double beta = constants().beta();

        double factor = rho_0*rho_0*g*beta;
        for( int i=0; i<ahh_vec.size(); i++ )
            ahh_vec[i] = rho_vec[i]*phi_vec[i]*dSw_vec[i] + rho_vec[i]*Sw_vec[i]*dphi_vec[i] + factor*phi_vec[i]*Sw_vec[i];
    }

    template <typename TVec>
    void VarSatPhysicsImpl<TVec>::initialise_shape_functions(const mesh::Mesh& m)
    {
        // matrices with weights for computing shape functions
        IntVector ia, ja;
        TVec shape_val, shape_dx, shape_dy, shape_dz;

        // Allocate row begin array
        int ia_length = m.interior_cvfaces() + 1;
        ia.resize(ia_length);

        // Fill row begin array
        ia[0] = 0;
        for (int i = 0; i < m.interior_cvfaces(); ++i) {
            ia[i+1] = ia[i] + m.cvface(i).element().nodes();
        }

        // Allocate matrix arrays
        int ja_length = ia[ia_length-1];
        ja.resize(ja_length);

        shape_val.resize(ja_length);
        shape_dx.resize(ja_length);
        shape_dy.resize(ja_length);
        shape_dz.resize(ja_length);

        // Allocate node value arrays
        h_vec.resize(m.nodes());

        // Allocate CVFace centroid arrays
        h_faces.resize(m.interior_cvfaces());
        grad_h_faces_.set(m.interior_cvfaces(), m.dim());

        // Fill other arrays;
        for (int i = 0; i < m.elements(); ++i) {

            const mesh::Element& e = m.element(i);

            // Sort the node ids, to get the index vector
            std::vector< std::pair<int, int> > index_vector(e.nodes());
            for (int k = 0; k < e.nodes(); ++k) {
                index_vector[k] = std::make_pair(e.node(k).id(), k);
            }
            std::sort(index_vector.begin(), index_vector.end());

            shape::Shape my_shape(e);
            for (int j = 0; j < e.edges(); ++j) {

                int cvf_id = e.cvface(j).id();

                // Record ja indices
                const mesh::CVFace& cvf = e.cvface(j);
                for (int k = 0, p = ia[cvf_id]; p < ia[cvf_id+1]; ++k, ++p) {
                    ja[p] = index_vector[k].first;
                }

                // Get shape functions and gradients
                std::vector<double> shape_functions = my_shape.shape_functions(j);
                std::vector<mesh::Point> shape_gradients = my_shape.shape_gradients(j);

                // Now load them into the matrices
                for (int k = 0, p = ia[cvf_id]; p < ia[cvf_id+1]; ++k, ++p) {
                    shape_val[p] = shape_functions[index_vector[k].second];
                    shape_dx[p]  = shape_gradients[index_vector[k].second].x;
                    shape_dy[p]  = shape_gradients[index_vector[k].second].y;
                    shape_dz[p]  = shape_gradients[index_vector[k].second].z;
                }
            }
        }

        shape_matrix = InterpolationMatrix(ia, ja, shape_val);
        shape_gradient_matrixX = InterpolationMatrix(ia, ja, shape_dx);
        shape_gradient_matrixY = InterpolationMatrix(ia, ja, shape_dy);
        if (dimension == 3)
            shape_gradient_matrixZ = InterpolationMatrix(ia, ja, shape_dz);

        // matrix for flux limitting
        IntVector ia_fl, ja_fl;
        TVec weights_fl;

        // allocate space for row begin indices
        ia_length = m.edges()+1;
        ia_fl.resize(ia_length);
        ia_fl[0] = 0;
        for (int i = 0; i < m.edges(); ++i) {
            ia_fl[i+1] = ia_fl[i] + m.edge_cvface(i).size();
        }

        // allocate space for column indices
        ja_length = ia_fl[ia_length-1];
        ja_fl.resize(ja_length);

        // allocate space for weights
        weights_fl.resize(ja_length);

        for(int i=0; i<m.edges(); i++){
            const std::vector<int>& faces = m.edge_cvface(i);

            // determine the total surface area of the faces attached to edge i
            double total_area = 0.;
            for(int j=0; j<faces.size(); j++)
                total_area += m.cvface(faces[j]).area();

            // now determine the scaled weights
            int pos = ia_fl[i];
            for(int j=0; j<faces.size(); j++){
                int face = faces[j];
                //weights_fl[pos] = m.cvface(face).area()/total_area;
                weights_fl[pos] = 1./total_area;
                ja_fl[pos] = face;
                pos++;
            }
        }

        flux_lim_matrix = InterpolationMatrix(ia_fl, ja_fl, weights_fl);
    }

    // get a copy of a set of physical zone properties
    template <typename TVec>
    const PhysicalZone& VarSatPhysicsImpl<TVec>::physical_zone( int zone ) const
    {
        if(!(zone>=0 && zone<physical_zones_.size()))
        assert(zone>=0 && zone<physical_zones_.size());
        return physical_zones_[zone];
    }

    // get the number of physical zones
    template <typename TVec>
    int VarSatPhysicsImpl<TVec>::physical_zones( void ) const
    {
        return physical_zones_.size();
    }

    template <typename TVec>
    const BoundaryCondition& VarSatPhysicsImpl<TVec>::boundary_condition_h( int tag ) const{
        std::map<int,BoundaryCondition>::const_iterator it = boundary_conditions_h_.find(tag);
        assert( it!=boundary_conditions_h_.end());
        return it->second;
    }
    */
} // end namespace fvmpor
