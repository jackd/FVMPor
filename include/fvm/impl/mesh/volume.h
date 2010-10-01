#ifndef MESH_VOLUME_H
#define MESH_VOLUME_H

namespace mesh {

class Volume {
public:
    Volume(const Mesh& mesh, int id);

    int id() const;
    const Mesh& mesh() const;
    const Node& node() const;

    int scvs() const;
    const SCV& scv(int i) const;

    double vol() const;
    Point centroid() const;

private:
    const Mesh* m;
    int my_id;
    double my_vol;
    Point my_centroid;
    std::vector<int> scvvec;

    friend class Mesh;
    void add_scv(int id);
    void compute_volume();
    void compute_centroid();
};

} // end namespace mesh

#endif
