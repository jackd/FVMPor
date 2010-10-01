#ifndef MESH_ELEMENT_H
#define MESH_ELEMENT_H

#include <utility>
#include <vector>
#include <ostream>

namespace mesh {

class Element {
public:
    Element();
//    Element(const Mesh& mesh, int id,
//            const std::vector<int>& nodevec,
//            const std::vector<int>& edgevec,
//            const std::vector<int>& facevec);
    Element(const Mesh& mesh, int type, int id,
            const std::vector<int>& nodevec,
            const std::vector<int>& edgevec,
            const std::vector<int>& facevec,
            int physical_tag);

    int id() const;
    int physical_tag() const;
    const Mesh& mesh() const;
    int type() const; // NEWMESH
    int nodes() const;
    int edges() const;
    int faces() const;
    int scvs() const;
    int cvfaces() const;
    int neighbours() const;
    int dim() const;

    const Node& node(int i) const;
    const Edge& edge(int i) const;
    const Face& face(int i) const;
    const SCV& scv(int i) const;
    const CVFace& cvface(int i) const;
    const Element& neighbour(int i) const;

    Point centroid() const;

    int node_id(int i) const;
    int edge_id(int i) const;
    int face_id(int i) const;

    int node_id(const Node&) const;
    int edge_id(const Edge&) const;
    int face_id(const Face&) const;

    std::pair<int, int> faces_from_edge(const Edge&) const;
    int face_from_edges(const Edge&, const Edge&) const;
    std::pair<int, int> other_edges_for_node(const Node&, const Edge&) const;
    int other_edge_for_node(const Node&, const Edge&) const;

    friend std::ostream &operator<<(std::ostream &cout, Element el);
private:
    const Mesh* m;
    int my_id;
    int my_physical_tag;
    int my_type; // NEWMESH
    std::vector<int> nodevec;
    std::vector<int> edgevec;
    std::vector<int> facevec;
    std::vector<int> scvvec;
    std::vector<int> cvfacevec;
    std::vector<int> neighbourvec;
    Point my_centroid;

    friend class Mesh;
    void add_scv(int id);
    void add_cvface(int id);
    void add_neighbour(int id);
};

} // end namespace mesh

#endif
