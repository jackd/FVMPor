#ifndef MESH_VOLUME_INLINE_H
#define MESH_VOLUME_INLINE_H

namespace mesh {

inline
const Mesh& Volume::mesh() const {
    return *m;
}

inline
int Volume::id() const {
    return my_id;
}

inline
const Node& Volume::node() const {
    return mesh().node(id());
}

inline
int Volume::scvs() const {
    return scvvec.size();
}

inline
const SCV& Volume::scv(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= scvs())
        throw OutOfRangeException("Volume::scv(int): out of range");
    #endif
    return mesh().scv(scvvec[i]);
}

inline
double Volume::vol() const {
    return my_vol;
}

inline
Point Volume::centroid() const {
    return my_centroid;
}

inline
void Volume::add_scv(int id) {
    scvvec.push_back(id);
}

inline
void Volume::compute_volume() {
    for (int i = 0; i < scvs(); ++i)
        my_vol += scv(i).vol();
}

inline
void Volume::compute_centroid() {
    for (int i = 0; i < scvs(); ++i)
        my_centroid += scv(i).vol() * scv(i).centroid();
    my_centroid /= vol();
}

inline
Volume::Volume(const Mesh& mesh, int id)
    : m(&mesh), my_id(id), my_vol(0.0) {}

} // end namespace mesh

#endif
