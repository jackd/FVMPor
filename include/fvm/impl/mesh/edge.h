#ifndef MESH_EDGE_H
#define MESH_EDGE_H

#include <functional>
#include <ostream>

namespace mesh {

class Edge {
public:
    Edge();
    Edge(const Mesh& mesh, int id, int front, int back);

    int id() const;
    const Mesh& mesh() const;
    const Node& front() const;
    const Node& back() const;

    Point midpoint() const;

    friend std::ostream &operator<<(std::ostream &cout, Edge ed);
private:
    friend class Mesh;

    const Mesh* m;
    int my_id;
    int front_id, back_id;
    Point my_midpoint;
};

bool operator<(const Edge&, const Edge&);

} // end namespace mesh

#endif
