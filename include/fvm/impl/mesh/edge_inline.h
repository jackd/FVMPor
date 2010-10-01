#ifndef MESH_EDGE_INLINE_H
#define MESH_EDGE_INLINE_H

#include <ostream>

namespace mesh {

inline
const Mesh& Edge::mesh() const {
    return *m;
}

inline
int Edge::id() const {
    return my_id;
}

inline
Point Edge::midpoint() const {
    return my_midpoint;
}

inline
const Node& Edge::front() const {
    return mesh().node(front_id);
}

inline
const Node& Edge::back() const {
    return mesh().node(back_id);
}

inline
bool operator<(const Edge& a, const Edge& b) {
    int amax = a.front().id();
    int amin = a.back().id();
    if (amin > amax)
        std::swap(amin, amax);

    int bmax = b.front().id();
    int bmin = b.back().id();
    if (bmin > bmax)
        std::swap(bmin, bmax);

    if (amin < bmin)
        return true;
    if (bmin < amin)
        return false;
    return (amax < bmax);
}

inline
Edge::Edge() : m(0), my_id(-1), front_id(-1), back_id(-1), my_midpoint() {}

inline
Edge::Edge(const Mesh& mesh, int id, int frontp, int backp)
    : m(&mesh), my_id(id), front_id(frontp), back_id(backp)
{
    my_midpoint = (front().point() + back().point()) / 2.0;
}

inline
std::ostream &operator<<(std::ostream &os, Edge ed)
{
    os << "Edge " << ed.id() << ", [front,back] = [" << ed.front().id() << ", " << ed.back().id() << "]" << std::endl;
    return os;
}

} // end namespace mesh

#endif
