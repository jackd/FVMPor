#ifndef MESH_NODE_H
#define MESH_NODE_H


#include <ostream>
#include <vector>

namespace mesh {



class Node {
public:
    Node();
    Node(const Mesh& mesh, int id, const std::vector<int>& boundary, Point p);

    int id() const;
    const Volume& volume() const;
    int boundaries() const;
    int boundary() const;
    int boundary(int) const; // pre: boundaries() != 0
    bool on_boundary( int ) const;
    const Mesh& mesh() const;
    Point point() const;

    friend std::ostream &operator<<(std::ostream &cout, Node n);
private:
    friend class Mesh;
    const Mesh* m;
    int my_id;
    std::vector<int> boundary_id;
    Point p;
};

bool operator<(const Node&, const Node&);

} // end namespace mesh

#endif
