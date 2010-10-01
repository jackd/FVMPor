#ifndef MESH_MESH_INLINE_H
#define MESH_MESH_INLINE_H

namespace mesh {

inline
const Pattern& Mesh::node_pattern() const{
    return node_pattern_;
}

inline 
int Mesh::domains() const {
    return n_dom;
}

inline 
mpi::MPICommPtr Mesh::mpicomm() const {
    return mpicomm_;
}

inline 
int Mesh::dim() const {
    return mesh_dim_;
}

inline 
int Mesh::domain_id() const {
    return dom_id;
}

inline
const std::vector<int>& Mesh::vtxdist() const {
    return vtx_dist;
}

inline
int Mesh::global_nodes() const {
    return n_nodes_gbl_;
}

inline
int Mesh::local_nodes() const {
    return n_nodes_loc_;
}

inline
int Mesh::internal_nodes() const {
    return n_nodes_int_;
}

inline
int Mesh::boundary_nodes() const {
    return n_nodes_bnd_;
}

inline
int Mesh::external_nodes() const {
    return n_nodes_ext_;
}

inline
int Mesh::nodes() const {
    return nodevec.size();
}

inline
int Mesh::local_elements() const {
    return n_elements_int;
}

inline
int Mesh::external_elements() const {
    return n_elements_bnd;
}

inline
int Mesh::elements() const {
    return elementvec.size();
}

inline
int Mesh::edges() const {
    return edgevec.size();
}

inline
int Mesh::interior_faces() const {
    return n_faces_int;
}

inline
int Mesh::boundary_faces() const {
    return n_faces_bnd;
}

inline
int Mesh::faces() const {
    return facevec.size();
}

inline
int Mesh::interior_cvfaces() const {
    return n_cvfaces_int;
}

inline
int Mesh::boundary_cvfaces() const {
    return n_cvfaces_bnd;
}

inline
int Mesh::cvfaces() const {
    return cvfacevec.size();
}

inline
int Mesh::scvs() const {
    return scvvec.size();
}

inline
int Mesh::external_node_id(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= external_nodes())
        throw OutOfRangeException("Mesh::external_node_id(int): out of range");
    #endif
    return nodes_ext[i];
}

inline
int Mesh::global_node_id(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= nodes())
        throw OutOfRangeException("Mesh::global_node_id(int): out of range");
    #endif
    if (i < local_nodes()) {
        return i + vtxdist()[domain_id()];
    } else {
        return external_node_id(i - local_nodes());
    }
}

inline
const Node& Mesh::node(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= nodes())
        throw OutOfRangeException("Mesh::node(int): out of range");
    #endif
    return nodevec[i];
}

inline
const Edge& Mesh::edge(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= edges())
        throw OutOfRangeException("Mesh::edge(int): out of range");
    #endif
    return edgevec[i];
}

inline
const Face& Mesh::face(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= faces())
        throw OutOfRangeException("Mesh::face(int): out of range");
    #endif
    return facevec[i];
}

inline
const Element& Mesh::element(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= elements())
        throw OutOfRangeException("Mesh::element(int): out of range");
    #endif
    return elementvec[i];
}

inline
const CVFace& Mesh::cvface(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= cvfaces())
        throw OutOfRangeException("Mesh::cvface(int): out of range");
    #endif
    return cvfacevec[i];
}

inline
const SCV& Mesh::scv(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= scvs())
        throw OutOfRangeException("Mesh::scv(int): out of range");
    #endif
    return scvvec[i];
}

inline
const Volume& Mesh::volume(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= nodes())
        throw OutOfRangeException("Mesh::volume(int): out of range");
    #endif
    return volumevec[i];
}

inline
int Mesh::boundaries() const {
    return boundary_tags.size() - 1;
}

inline
const std::vector<double>& Mesh::property(int i) const {
    #ifdef MESH_DEBUG
    //if (i < 0 || i >= elements())
    if (i < 0 || i >= num_properties())
        throw OutOfRangeException("Mesh::property(int): out of range");
    #endif
    //return properties[i];
    return properties[elementvec[i].physical_tag()];
}

inline
const std::vector<int>& Mesh::edge_cvface(int i) const{
    #ifdef MESH_DEBUG
    assert(i<edge_cvfaces.size());
    #endif
    return edge_cvfaces[i];
}

} // end namespace mesh

#endif
