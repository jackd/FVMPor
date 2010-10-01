#ifndef MESH_FACE_H
#define MESH_FACE_H

#include <functional>
#include <utility>
#include <vector>

namespace mesh {

class Face {
public:
    Face();
    Face(const Mesh& mesh, int id, int boundary,
         const std::vector<int>& nodevec,
         const std::vector<int>& edgevec);

    int id() const;
    int boundary() const;
    const Mesh& mesh() const;
    int nodes() const;
    int edges() const;
    const Node& node(int i) const;
    const Edge& edge(int i) const;
    Point centroid() const;

    static Face line(const Mesh& mesh, int id, int boundary,
                     int node0, int node1, int face0);

    static Face triangular(const Mesh& mesh, int id, int boundary,
                           int node0, int node1, int node2,
                           int face0, int face1, int face2);

    static Face rectangular(const Mesh& mesh, int id, int boundary,
                            int node0, int node1,
                            int node2, int node3,
                            int face0, int face1,
                            int face2, int face3);

    std::pair<int, int> edges_from_node(const Node&) const;

private:
    friend class Mesh;
    const Mesh* m;
    int my_id;
    int boundary_id;
    std::vector<int> nodevec;
    std::vector<int> edgevec;
    Point my_centroid;
    friend bool operator<(const Face&, const Face&);
};

} // end namespace mesh

#endif
