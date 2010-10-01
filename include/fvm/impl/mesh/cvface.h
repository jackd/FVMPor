#ifndef MESH_CVFACE_H
#define MESH_CVFACE_H

//#include "quad.h"
#include "cvface_shape.h"

namespace mesh {

class CVFace {
public:
    CVFace(const Mesh& mesh, int id,
           int element_id, int front_id,
           int back_id, int boundary_id,
           CVFace_shape shape, int tag, int edge_id);
    // The tag identifies which edge of the element the CV face bisects.  Thus
    // it can be used as in index into an array of shape functions.  Boundary
    // CV faces have tag == -1.
    
    int id() const;
    const Mesh& mesh() const;

    const Element& element() const;
    const Node& front() const;
    const Node& back() const;
    const Edge& edge() const;
    int boundary() const;
    int tag() const;

    double area() const;
    Point normal() const;
    Point unit_normal() const;
    Point centroid() const;

    Point point(int i) const;

private:
    friend class Mesh;
    const Mesh* m;
    int my_id;
    int element_id;
    int front_id, back_id;
    int boundary_id;
    int edge_id;
    //Quadrilateral q;
    CVFace_shape shape;
    int my_tag;
    double my_area;
    Point my_normal;
    Point my_centroid;
    friend bool operator<(const CVFace&, const CVFace&);
};

} // end namespace mesh

#endif
