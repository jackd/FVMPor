#ifndef MESH_CVFACE_INLINE_H
#define MESH_CVFACE_INLINE_H

#include <iostream>

namespace mesh {

inline
const Mesh& CVFace::mesh() const {
    return *m;
}

inline
int CVFace::id() const {
    return my_id;
}

inline
int CVFace::boundary() const {
    return boundary_id;
}

inline
int CVFace::tag() const {
    return my_tag;
}

inline
const Element& CVFace::element() const {
    return mesh().element(element_id);
}

inline
const Node& CVFace::front() const {
    if (boundary()) {
        throw OutOfRangeException("CVFace::front(): boundary()");
    }
    return mesh().node(front_id);
}

inline
const Node& CVFace::back() const {
    return mesh().node(back_id);
}

inline
Point CVFace::point(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= shape.points()) {
        throw OutOfRangeException("CVFace::point(int): out of range");
    }
    #endif
    return shape.point(i);
}
inline
double CVFace::area() const {
    return my_area;
}

inline
Point CVFace::normal() const {
    return my_normal * my_area;
}

inline
Point CVFace::unit_normal() const {
    return my_normal;
}

inline
Point CVFace::centroid() const {
    return my_centroid;
}

inline 
const Edge& CVFace::edge() const{
    // only internal CV faces have a unique edge
    // so throw an error if somebody tries to access the edge
    // of a boundary CV face
    assert(edge_id>=0);
    return mesh().edge(edge_id);
}

inline
bool operator<(const CVFace& a, const CVFace& b) {
    if (a.edge().id()<b.edge().id()) return true;
    if (a.edge().id()>b.edge().id()) return false;
    if (a.element_id < b.element_id) return true;
    return false;
    /*
    if (a.front_id < b.front_id) return true;
    if (a.front_id > b.front_id) return false;
    return a.back_id < b.back_id;
    */
}

inline
CVFace::CVFace(const Mesh& mesh, int id, int element,
               int front_id, int back_id, int boundary,
               CVFace_shape shape, int tag, int edge_id)
    : m(&mesh), my_id(id), element_id(element),
      front_id(front_id), back_id(back_id), boundary_id(boundary), edge_id(edge_id), shape(shape),
      my_tag(tag)
{
    my_area = shape.area();
    my_centroid = shape.centroid();
    my_normal = shape.normal();

    // Check if normal points outwards (i.e. points from back node to front node)
    // (Note: this check is only for non-boundary faces.  Boundary faces have
    // to be supplied in correct outwards orientation - no check is made.)
    if (!boundary) {
        Point outwards = front().point() - back().point();
        if (dot(my_normal, outwards) < 0.0) {
            // it doesn't, so fix the orientation and negate the normal
            shape.reverse();
            my_normal = -my_normal;
        }
        assert(dot(my_normal, outwards) > 0.0);
    }
}

} // end namespace mesh

#endif
