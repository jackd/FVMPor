#ifndef FVMPOR_H
#define FVMPOR_H

#include "definitions.h"
#include "shape.h"

#include <fvm/fvm.h>
#include <fvm/mesh.h>
#include <fvm/solver_compact.h>
#include <fvm/physics_base.h>

#include <util/intvector.h>
#include <util/interpolation.h>
#include <util/dimvector.h>
#include <util/timer.h>

#include <cublas.h>
#include <mkl_spblas.h>
#include <mkl_service.h>

#include <vector>
#include <memory>
#include <map>


namespace fvmpor {

template <typename T>
struct CoordTraits_{
    static bool is_device() {return false;};
};
template <>
struct CoordTraits_<lin::gpu::Coordinator<int> >{
    static bool is_device() {return true;};
};

enum SpatialWeightType {weightUpwind, weightAveraging, weightVanLeer};

using lin::all;

template <typename CoordHost, typename CoordDevice>
class DensityDrivenPhysicsImpl{
public:
    typedef typename lin::rebind<CoordHost, double>::type CoordHostDouble;
    typedef typename lin::rebind<CoordHost, int>::type CoordHostInt;
    typedef typename lin::rebind<CoordDevice, double>::type CoordDeviceDouble;
    typedef typename lin::rebind<CoordDevice, int>::type CoordDeviceInt;

    typedef lin::Vector<double, CoordHostDouble> TVec;
    typedef lin::Vector<int, CoordHostInt> TIndexVec;
    typedef lin::Vector<double, CoordDeviceDouble> TVecDevice;
    typedef lin::Vector<int, CoordDeviceInt> TIndexVecDevice;

    typedef util::InterpolationMatrix<CoordDevice> InterpolationMatrix;
    typedef util::DimVector<TVecDevice> DimVector;

protected:
    typedef mesh::Point Point;

    // computation during a timestep
    void process_faces_lim( const mesh::Mesh &m );
    void process_faces_shape( const mesh::Mesh &m );
    void process_volumes_psk( const mesh::Mesh &m );
    void process_derivative_coefficients( const mesh::Mesh &m );
    void process_fluxes( double t, const mesh::Mesh &m );
    void process_spatial_weights(const mesh::Mesh& m);

    // physical zones
    const PhysicalZone& physical_zone( int ) const;
    int physical_zones() const;

    // boundary conditions
    int boundary_conditions() const { return boundary_conditions_h_.size(); };
    const BoundaryCondition& boundary_condition_h( int ) const;
    const BoundaryCondition& boundary_condition_c( int ) const;
    const Constants& constants() const { return constants_; };

    ////////////////////////////////
    // routines for setting up
    ///////////////////////////////
    void set_physical_zones();
    void set_boundary_conditions();
    void initialise_vectors( const mesh::Mesh &m );
    void set_initial_conditions( double &t, const mesh::Mesh& m );
    void set_constants();
    void initialise_shape_functions(const mesh::Mesh& m);

    // communicator for global communication of doubles on the nodes
    mpi::Communicator<double> node_comm_;

    // physical definitions
    int dimension;
    std::vector<PhysicalZone> physical_zones_;
    std::map<int,BoundaryCondition> boundary_conditions_h_;
    std::map<int,BoundaryCondition> boundary_conditions_c_;
    Constants constants_;
    // tags whether a node is dirichlet
    TIndexVec is_dirichlet_h_vec_;
    TIndexVec is_dirichlet_c_vec_;
    TIndexVecDevice dirichlet_h_nodes_; // DEVICE
    TIndexVecDevice dirichlet_c_nodes_; // DEVICE
    TVecDevice dirichlet_h_; // DEVICE
    TVecDevice dirichlet_c_; // DEVICE
    TIndexVecDevice dirichlet_faces_; // DEVICE

    // associated with ASE BCs
    TIndexVecDevice ASE_faces_, ASE_tags_;
    TIndexVecDevice boundary_face_nodes_;
    TVecDevice ASE_concentrations_;

    TVecDevice cvface_areas_; // DEVICE

    // spatial weighting
    int CV_flux_comm_tag;
    SpatialWeightType spatial_weighting;

    TIndexVecDevice CV_up; // DEVICE
    TVecDevice CV_flux; // DEVICE
    TIndexVecDevice edge_up; // DEVICE
    TIndexVecDevice edge_down; // DEVICE
    TVecDevice edge_flux; // DEVICE

    // derived quantities
    std::vector<TVecDevice> head_scv;
    std::vector<TVecDevice> c_scv;
    std::vector<TVecDevice> phi_scv;
    std::vector<TIndexVecDevice> index_scv;
    std::vector<TVecDevice> weight_scv;
    std::map<int, int> zones_map_;

    // spatial weighting for CV faces
    std::vector<TIndexVecDevice> n_front_;
    std::vector<TIndexVecDevice> n_back_;
    std::vector<TIndexVecDevice> p_front_;
    std::vector<TIndexVecDevice> q_front_;
    std::vector<TIndexVecDevice> p_back_;
    std::vector<TIndexVecDevice> q_back_;
    TVecDevice edge_weight_front_;
    TVecDevice edge_weight_back_;
    TIndexVecDevice edge_node_front_;
    TIndexVecDevice edge_node_back_;

    TVecDevice M_flux_faces_;
    TVecDevice C_flux_faces_;

    // for interpolation from nodes to CV faces
    InterpolationMatrix shape_matrix;
    InterpolationMatrix shape_gradient_matrixX;
    InterpolationMatrix shape_gradient_matrixY;
    InterpolationMatrix shape_gradient_matrixZ;
    InterpolationMatrix flux_lim_matrix;
    InterpolationMatrix cvflux_matrix;
    InterpolationMatrix dirichlet_matrix;

    TVecDevice h_vec_, c_vec_; // head and concentration at the nodes
    TVecDevice cp_vec_; // concentration derivative at the nodes
    DimVector grad_h_faces_; // head gradient at CV faces
    DimVector grad_c_faces_; // head gradient at CV faces
    TVecDevice h_faces_, c_faces_; // head and concentration at CV faces
    TVecDevice qdotn_faces_; // volumetric fluid flux at CV faces
    TVecDevice qcdotn_faces_; // volumetric solute flux at CV faces

    // storing derived quantities averaged for each control volume
    TVecDevice rho_vec_, phi_vec_;
    // storing derived quantities at cv faces (using c and h values at faces)
    TVecDevice rho_faces_, phi_faces_;
    // storing upwinded/flux limitted values at cv faces
    TVecDevice rho_faces_lim_, c_faces_lim_;
    // storing coefficients for derivative terms
    // these are constants
    TVecDevice ahc_vec_;
    double ahh_, ach_, acc_;
    // storing values at faces
    DimVector K_faces_;
    TVecDevice Dm_faces_;
    DimVector norm_faces_;
    DimVector qsat_faces_;
};


template <typename value_type, typename CoordHost, typename CoordDevice>
class DensityDrivenPhysics :
    public fvm::PhysicsBase< DensityDrivenPhysics<value_type, CoordHost, CoordDevice>,
                             value_type>,
    public DensityDrivenPhysicsImpl<CoordHost,CoordDevice>
{
    typedef fvm::PhysicsBase<DensityDrivenPhysics, value_type> base;
    typedef DensityDrivenPhysicsImpl<CoordHost,CoordDevice> impl;

    int num_calls;
    friend class Preconditioner;

    typename impl::TVecDevice res_tmp;
    typename impl::TVec res_tmp_host;
public:

    typedef typename base::iterator iterator;
    typedef typename base::const_iterator const_iterator;
    typedef typename base::Callback Callback;

    DensityDrivenPhysics() : num_calls(0) {};
    int calls() const { return num_calls; }

    /////////////////////////////////
    // GLOBAL
    /////////////////////////////////
    value_type flux(double t, const mesh::CVFace& cvf, const_iterator sol) const;
    value_type boundary_flux(double t, const mesh::CVFace& cvf, const_iterator sol) const;

    double compute_mass(const mesh::Mesh& m, const_iterator u);
    double mass_flux_per_time(const mesh::Mesh& m);

    /////////////////////////////////
    // VARIABLE-SPECIFIC
    /////////////////////////////////
    void initialise(double& t, const mesh::Mesh& m, iterator u, iterator udash, iterator temp, Callback);
    void preprocess_evaluation(double t, const mesh::Mesh& m, const_iterator u, const_iterator udash);
    void preprocess_timestep(double t, const mesh::Mesh& m, const_iterator sol, const_iterator deriv);
    void residual_evaluation( double t, const mesh::Mesh& m,
                              const_iterator sol, const_iterator deriv, iterator res);
};

/**************************************************************************
 *                          IMPLEMENTATION                                *
 **************************************************************************/
    using mesh::Point;

    template <typename TVec>
    void density(TVec& c, TVec& rho, const Constants& constants)
    {
        double factor = constants.rho_0()*constants.eta();

        rho.at(all) = factor*c;
        rho += constants.rho_0();
    }

    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::initialise_vectors( const mesh::Mesh &m ){
        dimension = m.dim();

        node_comm_.set_pattern( "NP_double", m.node_pattern() );

        if(CoordTraits_<CoordDeviceInt>::is_device()){
            std::cout << "intialising cublas" << std::endl;
            assert( cublasInit() == CUBLAS_STATUS_SUCCESS );
            std::cout << "initialised" << std::endl;
        }

        // set physical properties
        set_constants();
        set_physical_zones();
        set_boundary_conditions();

        // initialise space for storing p-s-k values
        int N = m.nodes();
        rho_vec_ = TVecDevice(N);
        phi_vec_ = TVecDevice(N);

        rho_faces_lim_ = TVecDevice(m.interior_cvfaces());
        c_faces_lim_ = TVecDevice(m.interior_cvfaces());
        rho_faces_ = TVecDevice(m.interior_cvfaces());
        phi_faces_ = TVecDevice(m.interior_cvfaces());

        // spatial weightings
        CV_up = TIndexVecDevice(m.local_nodes());
        CV_flux = TVecDevice(m.nodes());
        CV_flux_comm_tag = node_comm_.vec_add(CV_flux.data());

        edge_up = TIndexVecDevice(m.edges());
        edge_down = TIndexVecDevice(m.edges());
        edge_flux = TVecDevice(m.edges());

        M_flux_faces_ = TVecDevice(m.cvfaces());
        C_flux_faces_ = TVecDevice(m.cvfaces());
        qdotn_faces_ = TVecDevice(m.cvfaces());
        qcdotn_faces_ = TVecDevice(m.cvfaces());

        // initialise space for derivative coefficients
        ahc_vec_ = TVecDevice(m.local_nodes());

        // tag dirichlet nodes
        // also tag nodes that lie on seepage faces
        // assumes that if there is more than one seepage face, they all have the same tag
        is_dirichlet_h_vec_ = TIndexVec(m.local_nodes());
        is_dirichlet_c_vec_ = TIndexVec(m.local_nodes());
        int num_dirichlet_h = 0;
        int num_dirichlet_c = 0;
        for( int i=0; i<m.local_nodes(); i++ ){
            const mesh::Node& n = m.node(i);
            // look for dirichlet tags attached to the node
            for( int j=0; j<n.boundaries(); j++ ){
                int tag = n.boundary(j);
                if( boundary_condition_h(tag).is_dirichlet() ){
                    is_dirichlet_h_vec_[i] = tag;
                    num_dirichlet_h++;
                }
                if( boundary_condition_c(tag).is_dirichlet() ){
                    is_dirichlet_c_vec_[i] = tag;
                    num_dirichlet_c++;
                }
            }
        }

        // make a list of the dirichlet nodes
        TIndexVec dirichlet_h_nodes(num_dirichlet_h);
        TIndexVec dirichlet_c_nodes(num_dirichlet_c);
        int count_h=0;
        int count_c=0;
        for(int i=0; i<m.local_nodes(); i++){
            if(is_dirichlet_h_vec_[i])
                dirichlet_h_nodes[count_h++] = i;
            if(is_dirichlet_c_vec_[i])
                dirichlet_c_nodes[count_c++] = i;
        }
        // copy to device
        dirichlet_h_nodes_ = dirichlet_h_nodes;
        dirichlet_c_nodes_ = dirichlet_c_nodes;

        // store the prescribed head values
        // currently this only works for time-invariant dirichlet values
        TVec dirichlet_h(num_dirichlet_h);
        for(int n=0; n<num_dirichlet_h; n++){
            double t=0.;
            int i = dirichlet_h_nodes[n];
            const BoundaryCondition& bc = boundary_condition_h(is_dirichlet_h_vec_[i]);
            // fixed dirichlet
            if( bc.type()==1 ){
                dirichlet_h[n] = bc.value(t);
            }
            else{
                double el = dimension == 2 ? m.node(i).point().y : m.node(i).point().z;
                if(bc.type()==4)
                    dirichlet_h[n] = bc.hydrostatic(t, el);
                else
                    assert(false);
                //else
                //    dirichlet_h[n] = bc.hydrostatic_shore(t, el);
            }
        }
        TVec dirichlet_c(num_dirichlet_c);
        for(int n=0; n<num_dirichlet_c; n++){
            double t=0.;
            int i = dirichlet_c_nodes[n];
            const BoundaryCondition& bc = boundary_condition_c(is_dirichlet_c_vec_[i]);
            // only fixed dirichlet value possible for concentration
            assert(bc.type()==1);
            dirichlet_c[n] = bc.value(t);
        }
        // copy to device
        dirichlet_h_ = dirichlet_h;
        dirichlet_c_ = dirichlet_c;

        // creat flat arrays with information about ASE boundary faces
        std::vector<int> boundary_face_nodes;
        std::vector<int> ASE_faces;
        std::vector<int> ASE_tags;
        std::vector<double> ASE_concentrations;
        for(int i=m.interior_cvfaces(); i<m.cvfaces(); i++){
            boundary_face_nodes.push_back( m.cvface(i).back().id() );
            int boundary_tag = m.cvface(i).boundary();
            if(boundary_condition_c(boundary_tag).type()==5){
                ASE_faces.push_back(i-m.interior_cvfaces());
                ASE_tags.push_back(boundary_tag);
                ASE_concentrations.push_back(
                        boundary_condition_c(boundary_tag).value(0.)
                );
            }
        }
        // list of all faces that have ASE solute exchange
        ASE_faces_ = TIndexVec(ASE_faces.begin(), ASE_faces.end());
        // the BC tag of each ASE face
        ASE_tags_ = TIndexVec(ASE_tags.begin(), ASE_tags.end());
        // the concentration associated with inflow at each ASE face
        ASE_concentrations_ = TVec( ASE_concentrations.begin(),
                                    ASE_concentrations.end() );

        // list of back nodes associated with each boundary CV face
        boundary_face_nodes_ = TIndexVec( boundary_face_nodes.begin(),
                                          boundary_face_nodes.end() );

        // initialise vectors used in calculating derived quantities such as saturation
        // allocate room for each of the arrays
        std::set<int> zones;
        for(int i=0; i<m.elements(); i++)
            zones.insert(m.element(i).physical_tag());
        int num_zones = zones.size();
        int indx=0;
        for( std::set<int>::iterator it=zones.begin(); it!=zones.end(); it++)
            zones_map_[*it] = indx++;

        // temp var
        std::vector< std::vector<double> > weight_scv_tmp;
        std::vector< std::vector<int> > index_scv_tmp;
        weight_scv_tmp.resize( num_zones );
        index_scv_tmp.resize( num_zones );
        std::vector<std::map<int,int> > nodes_idx;
        nodes_idx.resize(num_zones);
        // compile index and weight information mapping node information to scv information
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
                    weight_scv_tmp[j].push_back(weights[j]);
                    index_scv_tmp[j].push_back(i);
                    nodes_idx[j][i] = index_scv_tmp[j].size()-1;
                }
            }
        }
        // copy index and weight information to the device
        weight_scv.resize( num_zones );
        index_scv.resize( num_zones );
        for(int i=0; i<num_zones; i++){
            // temporary host vectors
            TVec w_tmp(weight_scv_tmp[i].begin(), weight_scv_tmp[i].end());
            TIndexVec i_tmp(index_scv_tmp[i].begin(), index_scv_tmp[i].end());
            // fast copy to device
            weight_scv[i] = w_tmp;
            index_scv[i] = i_tmp;
        }

        // allocate room for head values mapped onto SCVs
        head_scv.resize( num_zones );
        phi_scv.resize( num_zones );
        for(int i=0; i<num_zones; i++){
            head_scv[i] = TVecDevice( index_scv[i].size() );
            phi_scv[i] = TVecDevice( index_scv[i].size() );
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
                faceEdge_map_front[z].insert(
                        std::pair<int, std::pair<int, int> >( n, std::pair<int, int>(f, i) )
                );
                n = nodes_idx[z][bid];
                faceEdge_map_back[z].insert(
                        std::pair<int, std::pair<int, int> >( n, std::pair<int, int>(f, i))
                );
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
            std::vector<int> n_front;
            std::vector<int> p_front;
            std::vector<int> q_front;
            std::vector<int> n_back;
            std::vector<int> p_back;
            std::vector<int> q_back;
            int len = head_scv[z].dim();
            for(int i=0; i<len; i++){
                std::pair<idxTypeIt, idxTypeIt> rng = faceEdge_map_front[z].equal_range(i);
                for( idxTypeIt it=rng.first; it!=rng.second; ++it ){
                    n_front.push_back(i); // local node id
                    q_front.push_back(it->second.first); // global face index
                    p_front.push_back(it->second.second); // global edge index
                }
                rng = faceEdge_map_back[z].equal_range(i);
                for( idxTypeIt it=rng.first; it!=rng.second; ++it ){
                    n_back.push_back(i); // local node id
                    q_back.push_back(it->second.first); // global face index
                    p_back.push_back(it->second.second); // global edge index
                }
            }
            n_front_[z] = TIndexVec(n_front.begin(), n_front.end());
            p_front_[z] = TIndexVec(p_front.begin(), p_front.end());
            q_front_[z] = TIndexVec(q_front.begin(), q_front.end());
            n_back_[z] = TIndexVec(n_back.begin(), n_back.end());
            p_back_[z] = TIndexVec(p_back.begin(), p_back.end());
            q_back_[z] = TIndexVec(q_back.begin(), q_back.end());
        }

        edge_weight_front_ = TVecDevice(m.edges(), 0.5);
        edge_weight_back_ = TVecDevice(m.edges(), 0.5);
        TIndexVec edge_node_front(m.edges());
        TIndexVec edge_node_back(m.edges());
        for( int i=0; i<m.edges(); i++){
            edge_node_front[i] = m.edge(i).front().id();
            edge_node_back[i] = m.edge(i).back().id();
        }
        // copy onto device
        edge_node_front_ = edge_node_front;
        edge_node_back_  = edge_node_back;

        // create flat arrays with cv face areas
        TVec cvface_areas(m.cvfaces());
        for(int i=0; i<m.cvfaces(); i++)
            cvface_areas[i] = m.cvface(i).area();
        cvface_areas_ = cvface_areas;


        // initialise the shape functions
        initialise_shape_functions(m);

        // initialise flux vecs
        qsat_faces_.set(m.interior_cvfaces(), m.dim());

        norm_faces_.set(m.interior_cvfaces(), m.dim());
        TVec X(m.interior_cvfaces());
        TVec Y(m.interior_cvfaces());
        TVec Z(m.interior_cvfaces());
        for( int i=0; i<m.interior_cvfaces(); i++ ){
            Point nrm = m.cvface(i).normal();
            X[i] = nrm.x/cvface_areas[i];
            Y[i] = nrm.y/cvface_areas[i];
            if( m.dim()==3 )
                Z[i] = nrm.z/cvface_areas[i];
        }
        norm_faces_.x() = X;
        norm_faces_.y() = Y;
        if(m.dim()==3)
            norm_faces_.z() = Z;

        K_faces_.set(m.interior_cvfaces(), m.dim());
        for( int i=0; i<m.interior_cvfaces(); i++ ){
            int tag = m.cvface(i).element().physical_tag();
            X[i] = -physical_zone(tag).K_xx;
            Y[i] = -physical_zone(tag).K_yy;
            if( m.dim()==3 )
                Z[i] = -physical_zone(tag).K_zz;
        }
        K_faces_.x() = X;
        K_faces_.y() = Y;
        if(m.dim()==3)
            K_faces_.z() = Z;

        TVec Dm_faces(m.interior_cvfaces());
        for( int i=0; i<m.interior_cvfaces(); i++ ){
            int tag = m.cvface(i).element().physical_tag();
            Dm_faces[i] = physical_zone(tag).Dm;
        }
        Dm_faces_ = Dm_faces;
    }

    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::
        process_faces_shape( const mesh::Mesh &m )
    {
        //density(h_faces_, c_faces_, rho_faces_, constants());
        density( c_faces_, rho_faces_, constants());
    }

    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::
        process_faces_lim( const mesh::Mesh &m )
    {
        if(CoordTraits_<CoordDeviceInt>::is_device()){
            lin::gpu::collect_edges(
                          rho_vec_.data(), rho_faces_lim_.data(), m.edges(),
                          edge_weight_front_.data(), edge_weight_back_.data(),
                          edge_node_front_.data(), edge_node_back_.data(),
                          flux_lim_matrix.row_ptrs().data(),
                          flux_lim_matrix.col_indexes().data()
            );
            lin::gpu::collect_edges(
                          c_vec_.data(), c_faces_lim_.data(), m.edges(),
                          edge_weight_front_.data(), edge_weight_back_.data(),
                          edge_node_front_.data(), edge_node_back_.data(),
                          flux_lim_matrix.row_ptrs().data(),
                          flux_lim_matrix.col_indexes().data()
            );
        }
        else{
            const int *ia = flux_lim_matrix.row_ptrs().data();
            const int *ja = flux_lim_matrix.col_indexes().data();
            double *rho_face_ptr = rho_faces_lim_.data();
            double *c_face_ptr = c_faces_lim_.data();
            double rho_edge;
            double c_edge;
            int e;
            int j;
#pragma omp parallel for schedule(static) shared(rho_face_ptr, c_face_ptr, ja, ia) private(e, j, rho_edge, c_edge)
            for( e=0; e<m.edges(); e++ ){
                rho_edge =
                    rho_vec_.at(edge_node_back_[e])*edge_weight_back_.at(e)
                  + rho_vec_.at(edge_node_front_[e])*edge_weight_front_.at(e);
                c_edge =
                    c_vec_.at(edge_node_back_[e])*edge_weight_back_.at(e)
                  + c_vec_.at(edge_node_front_[e])*edge_weight_front_.at(e);
                for( j=ia[e]; j<ia[e+1]; j++){
                    rho_face_ptr[ja[j]] = rho_edge;
                    c_face_ptr[ja[j]] = c_edge;
                }
            }
        }
    }

    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::
        process_fluxes( double t, const mesh::Mesh &m )
    {
        // initialise the flux to zero
        qdotn_faces_.zero();
        qcdotn_faces_.zero();

        double factor = 1./constants().rho_0();
        int ifaces = m.interior_cvfaces();
        // compute the vector quantity q at each internal CV face
        qsat_faces_.x().at(all) = mul(grad_h_faces_.x(), K_faces_.x());
        if( m.dim()==2 ){
            qsat_faces_.y().at(all) = grad_h_faces_.y();
            qsat_faces_.y() += factor*rho_faces_;
            qsat_faces_.y() *= K_faces_.y();
        }else{
            qsat_faces_.y().at(all) = mul(grad_h_faces_.y(), K_faces_.y());
            qsat_faces_.z().at(all) = grad_h_faces_.z();
            qsat_faces_.z() += factor*rho_faces_;
            qsat_faces_.z() *= K_faces_.z();
        }

        qdotn_faces_.at(0,ifaces-1) = mul(norm_faces_.x(), qsat_faces_.x());
        qdotn_faces_.at(0,ifaces-1) += mul(norm_faces_.y(), qsat_faces_.y());
        if( m.dim()==3 ){
            qdotn_faces_.at(0,ifaces-1) += mul(norm_faces_.z(), qsat_faces_.z());
        }

        // find salt flux at faces
        qcdotn_faces_.at(0,ifaces-1) = mul(norm_faces_.x(), grad_c_faces_.x());
        qcdotn_faces_.at(0,ifaces-1) += mul(norm_faces_.y(), grad_c_faces_.y());
        if( m.dim()==3 ){
            qcdotn_faces_.at(0,ifaces-1) += mul( norm_faces_.z(),
                                                grad_c_faces_.z() );
        }

        qcdotn_faces_.at(0,ifaces-1) *= Dm_faces_;
        qcdotn_faces_.at(all) *= 0.35;
        //qcdotn_faces_.at(0,ifaces-1) *= phi_faces_; // need to find phi at faces

        // find the mass and solute flux at each interior CV face
        M_flux_faces_.at(0,ifaces-1) = mul( rho_faces_lim_.at(all),
                                            qdotn_faces_.at(0,ifaces-1) );
        C_flux_faces_.at(0,ifaces-1) = mul( c_faces_lim_.at(all),
                                            qdotn_faces_.at(0,ifaces-1) );
        C_flux_faces_.at(0,ifaces-1) -= qcdotn_faces_.at(0,ifaces-1);

        // loop over boundary faces and find fluid flux where explicitly given by BCs
        // temp vectors on host for computing the boundary fluxes
        int faces_bnd = m.boundary_cvfaces();
        TVec qdotn_faces_bnd(faces_bnd);
        TVec qcdotn_faces_bnd(faces_bnd);
        for( int i=0; i<faces_bnd; i++)
        {
            const mesh::CVFace& cvf = m.cvface(i+m.interior_cvfaces());

            int boundary_tag = cvf.boundary();
            const BoundaryCondition& BCh = boundary_condition_h( boundary_tag );
            const BoundaryCondition& BCc = boundary_condition_c( boundary_tag );

            switch( BCh.type() ){
                // prescribed flux
                case 3:
                    qdotn_faces_bnd[i] = BCh.value(t);
                    break;
                // prescribed directional flux
                case 6:
                    qdotn_faces_bnd[i] = BCh.flux( t, cvf.normal() )/m.cvface(i).area();
                    break;
                default:
                    break;

            }
            switch( BCc.type() ){
                // prescribed flux
                case 3:
                    qcdotn_faces_bnd[i] = BCc.value(t);
                    break;
                // prescribed directional flux
                case 6:
                    // we have to scale here because the cvf.normal() is scaled by cvface area
                    //qcdotn_faces_bnd[i] = BCc.flux( t, cvf.normal() )/cvface_areas_[i];
                    qcdotn_faces_bnd[i] = BCc.flux( t, cvf.normal() )/m.cvface(i).area();
                    break;
                default:
                    break;
            }
        }
        qdotn_faces_.at(ifaces,m.cvfaces()-1) = qdotn_faces_bnd;
        qcdotn_faces_.at(ifaces,m.cvfaces()-1) = qcdotn_faces_bnd;

        // determine flux over Dirichlet faces
        if(dirichlet_faces_.dim()){
            TVecDevice flux_tmp(dirichlet_faces_.dim());
            dirichlet_matrix.matvec( qdotn_faces_, flux_tmp );
            qdotn_faces_.at(dirichlet_faces_) = flux_tmp;
        }

        TVecDevice c_faces(m.boundary_cvfaces());
        TVecDevice rho_faces(m.boundary_cvfaces());

        // set the concentration at each face to be that of its back node
        c_faces.at(all) = c_vec_.at(boundary_face_nodes_);

        // determine the concentration at the ASE faces.
        // that is, for faces at which net flux is into the domain
        // choose the concentration provided by the ASE condition
        if(CoordTraits_<CoordDeviceInt>::is_device()){
            lin::gpu::set_ASE( ASE_faces_.data(),
                               //qdotn_faces_.data() + ifaces,
                               qdotn_faces_.at(ifaces,lin::end).data(),
                               ASE_concentrations_.data(),
                               c_faces.data(),
                               ASE_faces_.dim() );
        }else{
            int i, f;
            double *c_ptr = c_faces.data();
            double *ASE_c_ptr = ASE_concentrations_.data();
            double *q_ptr = qdotn_faces_.data();
            int *ASE_f_ptr = ASE_faces_.data();

#pragma omp parallel for schedule(static) shared(ifaces, c_ptr, ASE_c_ptr, ASE_f_ptr) private(i, f)
            for(i=0; i<ASE_faces_.dim(); i++){
                f = ASE_f_ptr[i];
                if(q_ptr[f+ifaces]<0.)
                    c_ptr[f] = ASE_c_ptr[i];
            }
        }

        // find the density at boundary CV faces
        density(c_faces, rho_faces, constants());

        // find the mass and solute fluxes over each boundary face
        M_flux_faces_.at(ifaces,lin::end) =
                                mul(rho_faces.at(all), qdotn_faces_.at(ifaces,lin::end));
        C_flux_faces_.at(ifaces,lin::end) =
                                mul(c_faces.at(all), qdotn_faces_.at(ifaces,lin::end));

        // scale fluxes by CV face areas
        M_flux_faces_ *= cvface_areas_;
        C_flux_faces_ *= cvface_areas_;
    }

    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::process_spatial_weights(const mesh::Mesh& m){

        // determine the flux over each edge
        flux_lim_matrix.matvec( qdotn_faces_.at(0, m.interior_cvfaces()-1), edge_flux );

        switch( spatial_weighting ){
            case weightAveraging :
                assert(false);
                break;
            ////////////////////////////////////////////////////////
            // the upwinding case is simple
            ////////////////////////////////////////////////////////
            case weightUpwind :
                if(CoordTraits_<CoordDeviceInt>::is_device()){
                    // NOTE :
                    // the front and back weights have been swapped here to account for the
                    // sign on the flux
                    lin::gpu::set_weights_upwind(
                            edge_flux.data(),
                            edge_weight_back_.data(),
                            edge_weight_front_.data(),
                            m.edges()
                    );
                }
                else{
                    for(int i=0; i<m.edges(); i++){
                        if( edge_flux.at(i)<0. ){
                            edge_weight_front_.at(i) = 1.;
                            edge_weight_back_.at(i) = 0.;
                        }
                        else{
                            edge_weight_front_.at(i) = 0.;
                            edge_weight_back_.at(i) = 1.;
                        }
                    }
                }
                break;
            ////////////////////////////////////////////////////////
            // the flux limitting case takes a bit more work
            ////////////////////////////////////////////////////////
            case weightVanLeer :
                assert(false);
                /*
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
                    CV_flux[n] -= qdotn_faces_[i];
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

                *node_comm_.mpicomm() << "DensityDrivenPhysicsImpl::process_spatial_weights : communicating 2up fluxes values accross subdomain boundaries" << std::endl;
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
                */
        }
    }

    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::process_volumes_psk( const mesh::Mesh &m )
    {
        double rho_0 = constants().rho_0();

        // zero out vectors of CV-averaged derived quantities
        phi_vec_.zero();

        // for each zone calucluate the scv-weighted derived quantities and add them to the appropriated CV-averaged vectors
        double T=0.;
        for( std::map<int, int>::iterator it=zones_map_.begin(); it!=zones_map_.end(); it++){
            int zone = (*it).second;
            int indx = (*it).first;
            int n = index_scv.size();
            const PhysicalZone& props = physical_zone(indx);

            // get head data for this zone type
            head_scv[zone].at(all) = h_vec_.at(index_scv[zone]);

            phi_scv[zone].at(all) = props.phi;
            // copy into global vector
            phi_vec_.at(index_scv[zone]) += mul(phi_scv[zone], weight_scv[zone]);
        }

        // find the CV-averaged density - this is much simpler because density is not dependant on material properties
        // of the porous medium
        //density(h_vec_, c_vec_, rho_vec_, constants());
        density(c_vec_, rho_vec_, constants());
    }

    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::process_derivative_coefficients( const mesh::Mesh &m )
    {
        ahh_ = 0.;
        //std::cerr << ahc_vec_.dim() << " AND " << phi_vec_.dim() << std::endl;
        ahc_vec_.at(all) = phi_vec_.at(0,m.local_nodes()-1);
        double factor = constants().rho_0() * constants().eta();
        ahc_vec_ *= factor;
        ach_ = 0.;
        acc_ = 1.;
    }

    template <typename CoordHost, typename CoordDevice>
    void DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::initialise_shape_functions(const mesh::Mesh& m)
    {
        // matrices with weights for computing shape functions
        TIndexVec ia, ja;
        TVec shape_val, shape_dx, shape_dy, shape_dz;

        // Allocate row begin array
        int ia_length = m.interior_cvfaces() + 1;
        ia = TIndexVec(ia_length);

        // Fill row begin array
        ia[0] = 0;
        for (int i = 0; i < m.interior_cvfaces(); ++i) {
            ia[i+1] = ia[i] + m.cvface(i).element().nodes();
        }

        // Allocate matrix arrays
        int ja_length = ia[ia_length-1];
        ja = TIndexVec(ja_length);

        shape_val = TVec(ja_length);
        shape_dx = TVec(ja_length);
        shape_dy = TVec(ja_length);
        shape_dz = TVec(ja_length);

        // Allocate node value arrays
        h_vec_ = TVecDevice(m.nodes());
        c_vec_ = TVecDevice(m.nodes());
        cp_vec_ = TVecDevice(m.nodes());

        // Allocate CVFace centroid arrays
        h_faces_ = TVecDevice(m.interior_cvfaces());
        c_faces_ = TVecDevice(m.interior_cvfaces());
        grad_h_faces_.set(m.interior_cvfaces(), m.dim());
        grad_c_faces_.set(m.interior_cvfaces(), m.dim());

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

        //////////////////////////////////////////////////////////
        // MATRIX FOR FLUX LIMITTING
        // num_edges X num_cvfaces
        // sums the fluxes at each face associated with an edge
        // which gives the total flux between the control volumes
        // that share the edge
        //////////////////////////////////////////////////////////
        TIndexVec ia_fl, ja_fl;
        TVec weights_fl;

        // allocate space for row begin indices
        ia_length = m.edges()+1;
        ia_fl = TIndexVec(ia_length);
        ia_fl[0] = 0;
        for (int i = 0; i < m.edges(); ++i) {
            ia_fl[i+1] = ia_fl[i] + m.edge_cvface(i).size();
        }

        // allocate space for column indices
        ja_length = ia_fl[ia_length-1];
        ja_fl = TIndexVec(ja_length);

        // allocate space for weights
        //weights_fl.resize(ja_length);
        weights_fl = TVec(ja_length, 0.);

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
                weights_fl.at(pos) = 1./total_area;
                ja_fl[pos] = face;
                pos++;
            }
        }

        flux_lim_matrix = InterpolationMatrix(ia_fl, ja_fl, weights_fl);

        //////////////////////////////////////////////////////////
        // MATRIX FOR CALCULATING FLUX OVER A CV SURFACE
        // num_nodes X num_cvfaces
        // sums the flux over each CV face that defines the surface
        // of the control volume around each node
        //////////////////////////////////////////////////////////
        TIndexVec ia_cl, ja_cl;
        TVec weights_cl;
        int N=m.local_nodes();

        ia_length = N+1;
        ia_cl = TIndexVec(ia_length);
        ia_cl[0] = 0;
        TIndexVec face_counts(N,0);
        std::vector<int> col_indexes;
        std::vector<double> weights_tmp;
        for (int i = 0; i < N; ++i) {
            const mesh::Volume& v = m.volume(i);
            double w = 1./v.vol();
            std::vector<int> node_faces;
            // make a list of the cv faces that form the
            // surface of the control volume around node i
            for(int j=0; j<v.scvs(); j++){
                const mesh::SCV& s = v.scv(j);
                for(int k=0; k<s.cvfaces(); k++)
                    node_faces.push_back(s.cvface(k).id());
            }
            // sort the faces in ascending order
            std::sort(node_faces.begin(),node_faces.end());
            // add them to the column index
            for(int j=0; j<node_faces.size(); j++)
                col_indexes.push_back(node_faces[j]);
            // update the row pointer
            ia_cl[i+1] = ia_cl[i]+node_faces.size();
            // choose the weight for each face
            for(int j=0; j<node_faces.size(); j++){
                // note that the order of evaluation is very important here
                // because if a cv face lies on the boundary it
                // has no front node
                if(i==m.cvface(node_faces[j]).back().id())
                    weights_tmp.push_back(-w);
                else
                    weights_tmp.push_back(w);
            }
        }
        // assign the column index and weights
        ja_cl.assign(col_indexes.begin(), col_indexes.end());
        weights_cl.assign(weights_tmp.begin(), weights_tmp.end());

        cvflux_matrix = InterpolationMatrix(ia_cl, ja_cl, weights_cl);

        //////////////////////////////////////////////////////////
        // MATRIX FOR CALCULATING FLUX OVER DIRICHLET FACES
        // ? X ?
        //////////////////////////////////////////////////////////
        TIndexVec ia_bnd, ja_bnd;
        TVec weights_bnd;

        // make a list of all the dirichlet faces
        std::vector<int> dirichlet_faces;
        for( int i=m.interior_cvfaces(); i<m.cvfaces(); i++){
            if( boundary_condition_h( m.cvface(i).boundary() ).is_dirichlet() )
                dirichlet_faces.push_back(i);
        }
        dirichlet_faces_ = TIndexVec(dirichlet_faces.begin(), dirichlet_faces.end());
        ia_bnd = TIndexVec(dirichlet_faces.size()+1);
        ia_bnd[0] = 0;

        if(dirichlet_faces.size()){
            std::vector<int> ja_tmp;
            std::vector<double> w_tmp;
            for( int i=0; i<dirichlet_faces.size(); i++)
            {
                int f = dirichlet_faces[i];
                const mesh::CVFace& cvf = m.cvface(f);

                double boundary_area = 0.0;
                double other_area = 0.0;
                const mesh::Volume& v = cvf.back().volume();
                for (int ii = 0; ii < v.scvs(); ++ii) {
                    const mesh::SCV& scv = v.scv(ii);
                    for (int j = 0; j < scv.cvfaces(); ++j) {
                        const mesh::CVFace& subcvf = scv.cvface(j);
                        if( subcvf.id()>=m.interior_cvfaces() && boundary_condition_h(subcvf.boundary()).is_dirichlet() )
                            boundary_area += m.cvface(subcvf.id()).area();
                    }
                }
                for (int ii = 0; ii < v.scvs(); ++ii) {
                    const mesh::SCV& scv = v.scv(ii);
                    for (int j = 0; j < scv.cvfaces(); ++j) {
                        const mesh::CVFace& subcvf = scv.cvface(j);
                        int idx = subcvf.id();
                        if( idx<m.interior_cvfaces() || !boundary_condition_h(subcvf.boundary()).is_dirichlet() ){
                            int sign = subcvf.back().id() == cvf.back().id() ? -1 : 1;
                            w_tmp.push_back(sign*m.cvface(idx).area()/boundary_area);
                            ja_tmp.push_back(idx);
                        }
                    }
                }
                ia_bnd[i+1] = ja_tmp.size();
            }
            ja_bnd.assign(ja_tmp.begin(), ja_tmp.end());
            weights_bnd.assign(w_tmp.begin(), w_tmp.end());

            dirichlet_matrix = InterpolationMatrix(ia_bnd, ja_bnd, weights_bnd);
        }
    }

    // get a copy of a set of physical zone properties
    template <typename CoordHost, typename CoordDevice>
    const PhysicalZone& DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::physical_zone( int zone ) const
    {
        if(!(zone>=0 && zone<physical_zones_.size()))
        assert(zone>=0 && zone<physical_zones_.size());
        return physical_zones_[zone];
    }

    // get the number of physical zones
    template <typename CoordHost, typename CoordDevice>
    int DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::physical_zones( void ) const
    {
        return physical_zones_.size();
    }

    template <typename CoordHost, typename CoordDevice>
    const BoundaryCondition& DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::boundary_condition_h( int tag ) const{
        std::map<int,BoundaryCondition>::const_iterator it = boundary_conditions_h_.find(tag);
        assert( it!=boundary_conditions_h_.end());
        return it->second;
    }
    template <typename CoordHost, typename CoordDevice>
    const BoundaryCondition& DensityDrivenPhysicsImpl<CoordHost,CoordDevice>::boundary_condition_c( int tag ) const{
        std::map<int,BoundaryCondition>::const_iterator it = boundary_conditions_c_.find(tag);
        assert( it!=boundary_conditions_c_.end());
        return it->second;
    }
} // end namespace fvmpor

#endif
