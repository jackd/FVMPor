#ifndef MESH_NODE_INLINE_H
#define MESH_NODE_INLINE_H

#include <ostream>
#include <algorithm>

namespace mesh {

inline
const Mesh& Node::mesh() const {
    return *m;
}

inline
int Node::id() const {
    return my_id;
}

inline
const Volume& Node::volume() const {
    return mesh().volume(my_id);
}

inline
int Node::boundaries() const {
    //if (boundary_id[0] == 0) return 0;
    return boundary_id.size();
}

inline
int Node::boundary() const {
    //return boundary_id[0];
    return boundary_id.size();
}

inline
int Node::boundary( int i ) const {
    #ifdef MESH_DEBUG
    if (i >= boundaries()) {
        throw OutOfRangeException("Node::boundary(int): i >= boundaries()");
    }
    #endif
    return boundary_id[i];
}

inline
Point Node::point() const {
    return p;
}

inline
//Node::Node() : m(0), my_id(-1), boundary_id(1,0), p() {}
Node::Node() : m(0), my_id(-1), boundary_id(0), p() {}

inline
bool operator<(const Node& a, const Node& b) {
    if (a.id() < b.id())
        return true;
    else
        return false;
}

inline
bool Node::on_boundary( int tag ) const {
    return( std::find( boundary_id.begin(), boundary_id.end(), tag )!=boundary_id.end() );
}

inline
Node::Node(const Mesh& mesh, int id, const std::vector<int>& boundary, Point point)
    : m(&mesh), my_id(id), boundary_id(boundary), p(point)
{
      if(!boundary_id.empty())
          std::sort(boundary_id.begin(), boundary_id.end());
}


inline
std::ostream &operator<<(std::ostream &os, Node n)
{
    os << "Node " << n.id() << ": " << n.point() << ",\twith bcs: " << n.boundary() << "\t[ ";
    os << "\tBCs : ";
    for(int i=0; i<n.boundaries(); i++)
        os << n.boundary(i) << " ";
    os << "]";
    return os;
}

} // end namespace mesh

#endif
