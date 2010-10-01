#ifndef MESH_FACE_INLINE_H
#define MESH_FACE_INLINE_H

#include <algorithm>
#include <iterator>

namespace mesh {

inline
const Mesh& Face::mesh() const {
    return *m;
}

inline
int Face::id() const {
    return my_id;
}

inline
int Face::boundary() const {
    return boundary_id;
}

inline
int Face::nodes() const {
    return nodevec.size();
}

inline
int Face::edges() const {
    return edgevec.size();
}

inline
const Node& Face::node(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= nodes())
        throw OutOfRangeException("Face::node(int): out of range");
    #endif
    return mesh().node(nodevec[i]);
}

inline
const Edge& Face::edge(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= edges())
        throw OutOfRangeException("Face::edge(int): out of range");
    #endif
    return mesh().edge(edgevec[i]);
}

inline
Point Face::centroid() const {
    return my_centroid;
}

inline
bool operator<(const Face& a, const Face& b) {
    if (a.nodes() < b.nodes()) return true;
    if (a.nodes() > b.nodes()) return false;
    std::vector<int> anodes = a.nodevec;
    std::vector<int> bnodes = b.nodevec;
    std::sort(anodes.begin(), anodes.end());
    std::sort(bnodes.begin(), bnodes.end());
    return anodes < bnodes;
}

inline
Face::Face() : m(0), my_id(-1), boundary_id(0), my_centroid() {}

inline
Face::Face(const Mesh& mesh, int id, int boundary,
           const std::vector<int>& nodevec,
           const std::vector<int>& edgevec)
    : m(&mesh), my_id(id), boundary_id(boundary),
      nodevec(nodevec), edgevec(edgevec), my_centroid()
{
    for (int i = 0; i < nodes(); ++i)
        my_centroid += node(i).point();
    my_centroid /= nodes();
}

inline
Face Face::line(const Mesh& mesh, int id, int boundary, int node0, int node1,int edge0) {
    std::vector<int> node_ids(2);
    node_ids[0] = node0;
    node_ids[1] = node1;
    std::vector<int> edge_ids(1);
    edge_ids[0] = edge0;
    return Face(mesh, id, boundary, node_ids, edge_ids);
}

inline
Face Face::triangular(const Mesh& mesh, int id, int boundary,
                      int node0, int node1, int node2,
                      int edge0, int edge1, int edge2) {
    std::vector<int> node_ids(3);
    node_ids[0] = node0;
    node_ids[1] = node1;
    node_ids[2] = node2;
    std::vector<int> edge_ids(3);
    edge_ids[0] = edge0;
    edge_ids[1] = edge1;
    edge_ids[2] = edge2;
    return Face(mesh, id, boundary, node_ids, edge_ids);
}

inline
Face Face::rectangular(const Mesh& mesh, int id, int boundary,
                       int node0, int node1,
                       int node2, int node3,
                       int edge0, int edge1,
                       int edge2, int edge3) {
    std::vector<int> node_ids(4);
    node_ids[0] = node0;
    node_ids[1] = node1;
    node_ids[2] = node2;
    node_ids[3] = node3;
    std::vector<int> edge_ids(4);
    edge_ids[0] = edge0;
    edge_ids[1] = edge1;
    edge_ids[2] = edge2;
    edge_ids[3] = edge3;
    return Face(mesh, id, boundary, node_ids, edge_ids);
}

} // end namespace mesh

#endif
