#ifndef MESH_MESH_H
#define MESH_MESH_H

#include <mpi/mpicomm.h>
#include <fvm/impl/communicators/pattern.h>

#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace mesh {

class Mesh {
public:
    Mesh(const std::string& meshname, mpi::MPICommPtr comm);
private:
    Mesh(const Mesh&);
    Mesh& operator=(const Mesh&);
public:
    // the MPI communicator
    mpi::MPICommPtr mpicomm() const;

    int domains() const;
    // number of domains in mesh

    int domain_id() const;
    // id of this domain

    const std::vector<int>& vtxdist() const;
    // vertex distribution vector [v_0, v_1, ..., v_n]
    // where process P_i is responsible for vertices v_i ... v_{i+1}-1

    int global_nodes() const;
    // number of nodes in *entire mesh*

    int local_nodes() const;
    // number of nodes local to this domain

    int external_nodes() const;
    // number of external nodes that connect to local nodes

    int boundary_nodes() const;
    // number of external nodes that connect to local nodes

    int internal_nodes() const;
    // number of external nodes that connect to local nodes

    int nodes() const;
    // number of domain-related nodes == local_nodes() + external_nodes()

    int local_elements() const;
    // number of elements that involve only local nodes

    int external_elements() const;
    // number of elements that involve some local and some external nodes

    int elements() const;
    // number of domain elements == local_elements() + external_elements()

    int edges() const;
    // number of edges for this domain

    int interior_faces() const;
    // number of interior faces for this domain

    int boundary_faces() const;
    // number of boundary faces for this domain (i.e. faces on a boundary)

    int faces() const;
    // number of faces for this domain == internal_faces() + boundary_faces()

    int interior_cvfaces() const;
    // number of interior CV faces for this domain

    int boundary_cvfaces() const;
    // number of boundary CV faces for this domain (i.e. CV faces on a boundary)

    int cvfaces() const;
    // number of CV faces for this domain == interior_cvfaces() + boundary_cvfaces()

    int scvs() const;
    // number of sub control volumes for this domain

    int global_node_id(int i) const;
    // global id of ith node

    int external_node_id(int i) const;
    // global id of ith external node

    const Node& node(int i) const;
    // ith domain node
    // pre: i in [0, nodes())
    // notes: i in [0, local_nodes()) represents a local node
    //        i in [local_nodes(), nodes()) represents an external node

    const Edge& edge(int i) const;
    // ith domain edge
    // pre: i in [0, edges())

    const Face& face(int i) const;
    // ith domain face
    // pre: i in [0, faces())
    // notes: i in [0, interior_faces()) represents an interior face
    //        i in [interior_faces, faces()) represents a boundary face

    const Element& element(int i) const;
    // ith domain element
    // pre: i in [0, elements())
    // notes: i in [0, local_elements()) represents a local element
    //        i in [local_elements(), elements()) represents an external element

    const CVFace& cvface(int i) const;
    // ith domain CV face
    // pre: i in [0, cvfaces())
    // notes: i in [0, interior_cvfaces()) represents an interior CV face
    //        i in [interior_cvfaces(), cvfaces()) represents a boundary CV face

    const SCV& scv(int i) const;
    // ith domain sub control volume
    // pre: i in [0, scvs())

    const Volume& volume(int i) const;
    // ith domain control volume
    // pre: i in [0, nodes())

    double local_vol() const;
    // volume of domain control volumes

    double vol() const;
    // volume of domain elements

    double total_vol() const;
    // volume of entire mesh

    int boundaries() const;
    // number of distinct boundary tags, excluding zero

    const std::vector<double>& property(int i) const;
    // ith element property vector

    int num_properties() const; 
    // number of material properties in the mesh

    int dim() const; 
    // dimsension of the mesh (=2 triangles etc) (=3 tets etc)

    // number of distinct physical properties
    const std::vector<int>& edge_cvface(int i) const;

    const Pattern& node_pattern() const;
private:
    mpi::MPICommPtr mpicomm_;

    int mesh_dim_;
    int n_dom, dom_id;
    int n_nodes_gbl_; // 
    int n_nodes_loc_; // number of nodes local to the domain
    int n_nodes_int_; // number of local nodes that ARE NOT external nodes for another domain
    int n_nodes_bnd_; // number of local nodes that ARE external nodes for another domain
    int n_nodes_ext_; // number of nodes that are local to another domain
    int n_elements_int, n_elements_bnd;
    int n_faces_int, n_faces_bnd;
    int n_cvfaces_int, n_cvfaces_bnd;
    int n_physical_props;
    std::vector<int> vtx_dist;
    std::vector<int> nodes_ext;
    std::vector<Node> nodevec;
    std::vector<Edge> edgevec;
    std::vector<Face> facevec;
    std::vector<Element> elementvec;
    std::vector<CVFace> cvfacevec;
    std::vector<SCV> scvvec;
    std::vector<Volume> volumevec;
    std::set<int> boundary_tags;
    std::vector< std::vector<double> > properties;
    std::vector< std::vector<int> > edge_cvfaces;
    Pattern node_pattern_;

    void open_mesh_file(const std::string&, std::ifstream&, std::ifstream&);
    void read_mesh_data(std::ifstream&, std::ifstream&);
    void read_header_data(std::ifstream&, int&, int&, int&);
    void read_properties(std::ifstream&, int);
    void read_external_nodes(std::ifstream&, int);
    void read_nodes(std::ifstream&, int);
    void read_elements(std::ifstream&, int,
        std::set<Edge>&, std::set<Face>&);
    void construct_edges_and_faces(int,
        const std::vector<int>&, const std::vector<int>&,
        std::vector<int>&, std::vector<int>&,
        std::set<Edge>&, std::set<Face>&);
    void set_element_neighbours();
    void populate_edges_and_faces(const std::set<Edge>&, const std::set<Face>&);
    void face_edge_sanity_check();
    void element_cvface_sanity_check();
    void boundary_tag_sanity_check();
    void construct_control_volumes();
    void initialise_volumes_and_faces();
    void construct_scv_faces_internal_3D();
    void construct_scv_faces_boundary_3D();
    void construct_scv_faces_internal_2D();
    void construct_scv_faces_boundary_2D();
    void construct_volumes();
    void construct_node_pattern();
    int insert_edge(std::set<Edge>&, const Edge&);
    int insert_face(std::set<Face>&, const Face&);
    void reorder_nodes_edges(std::set<Edge> &edgeset, std::set<Face> &faceset);
};

} // end namespace mesh

#endif
