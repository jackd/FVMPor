#ifndef MESH_SCV_H
#define MESH_SCV_H

namespace mesh {

class SCV {
public:
    SCV(const Mesh& mesh, int id, int element, int node);

    int id() const;
    const Mesh& mesh() const;

    const Element& element() const;
    const Node& node() const;

    int cvfaces() const;
    int boundary_cvfaces() const;
    const CVFace& cvface(int i) const;

    int vertices() const;
    Point vertex(int i) const;
    double vol() const;
    Point centroid() const;

private:
    const Mesh* m;
    int my_id;
    int element_id;
    int node_id;
    int boundary_faces;
    double my_vol;
    std::vector<int> cvfacevec;
    std::vector<Point> vertexvec;
    Point c;

    friend class Mesh;
    void add_cvface(int id);
    void add_boundary_cvface(int id);
    void add_volume(double vol);
    void set_vertices2D(Point p1, Point p2, Point p3, Point p4);
    void set_vertices3D(Point p1, Point p2, Point p3, Point p4,
                      Point p5, Point p6, Point p7, Point p8);
    void set_centroid(Point c);
};

} // end namespace mesh

#endif
