#ifndef MESH_SCV_INLINE_H
#define MESH_SCV_INLINE_H

namespace mesh {

inline
const Mesh& SCV::mesh() const {
    return *m;
}

inline
int SCV::id() const {
    return my_id;
}

inline
const Element& SCV::element() const {
    return mesh().element(element_id);
}

inline
const Node& SCV::node() const {
    return mesh().node(node_id);
}

inline
int SCV::cvfaces() const {
    return cvfacevec.size();
}

inline
int SCV::boundary_cvfaces() const {
    return boundary_faces;
}

inline
int SCV::vertices() const {
    return vertexvec.size();
}

inline
const CVFace& SCV::cvface(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= cvfaces())
        throw OutOfRangeException("SCV::cvface(int): out of range");
    #endif
    return mesh().cvface(cvfacevec[i]);
}

inline
Point SCV::vertex(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= vertices()) {
        throw OutOfRangeException("SCV::vertex(int): out of range");
}
    #endif
    return vertexvec[i];
}

inline
double SCV::vol() const {
    return my_vol;
}

inline
Point SCV::centroid() const {
    return c;
}

inline
void SCV::add_cvface(int id) {
    cvfacevec.push_back(id);
}

inline
void SCV::add_boundary_cvface(int id) {
    cvfacevec.push_back(id);
    ++boundary_faces;
}

inline
void SCV::add_volume(double vol) {
    my_vol += vol;
}

inline
void SCV::set_vertices2D(Point p1, Point p2, Point p3, Point p4) {
    if (vertexvec.empty()) {
        vertexvec.push_back(p1);
        vertexvec.push_back(p2);
        vertexvec.push_back(p3);
        vertexvec.push_back(p4);
    }
}

inline
void SCV::set_vertices3D(Point p1, Point p2, Point p3, Point p4, Point p5, Point p6, Point p7, Point p8) {
    if (vertexvec.empty()) {
        vertexvec.push_back(p1);
        vertexvec.push_back(p2);
        vertexvec.push_back(p3);
        vertexvec.push_back(p4);
        vertexvec.push_back(p5);
        vertexvec.push_back(p6);
        vertexvec.push_back(p7);
        vertexvec.push_back(p8);
    }
}

inline
void SCV::set_centroid(Point centroid) {
    c = centroid;
}

inline
SCV::SCV(const Mesh& mesh, int id, int element, int node)
    : m(&mesh), my_id(id), element_id(element), node_id(node),
      boundary_faces(0), my_vol(0.0)
{}

} // end namespace mesh

#endif
