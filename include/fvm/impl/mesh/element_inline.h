#ifndef MESH_ELEMENT_INLINE_H
#define MESH_ELEMENT_INLINE_H

#include <ostream>

namespace mesh {

inline
const Mesh& Element::mesh() const {
    return *m;
}

inline
int Element::id() const {
    return my_id;
}

inline
int Element::type() const {
    return my_type;
}

inline
int Element::physical_tag() const {
    return my_physical_tag;
}

inline
int Element::nodes() const {
    return nodevec.size();
}

inline
int Element::edges() const {
    return edgevec.size();
}

inline
int Element::faces() const {
    return facevec.size();
}

inline
int Element::scvs() const {
    return scvvec.size();
}

inline
int Element::cvfaces() const {
    return cvfacevec.size();
}

inline
int Element::neighbours() const {
    return neighbourvec.size();
}

inline
int Element::dim() const {
    if( type()<4 )
        return 2;
    return 3;
}

inline
const Node& Element::node(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= nodes())
        throw OutOfRangeException("Element::node(int): out of range");
    #endif
    return mesh().node(nodevec[i]);
}

inline
const Edge& Element::edge(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= edges())
        throw OutOfRangeException("Element::edge(int): out of range");
    #endif
    return mesh().edge(edgevec[i]);
}

inline
const Face& Element::face(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= faces())
        throw OutOfRangeException("Element::face(int): out of range");
    #endif
    return mesh().face(facevec[i]);
}

inline
const SCV& Element::scv(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= scvs())
        throw OutOfRangeException("Element::scv(int): out of range");
    #endif
    return mesh().scv(scvvec[i]);
}

inline
const CVFace& Element::cvface(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= cvfaces())
        throw OutOfRangeException("Element::cvface(int): out of range");
    #endif
    return mesh().cvface(cvfacevec[i]);
}

inline
const Element& Element::neighbour(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= neighbours())
        throw OutOfRangeException("Element::neighbour(int): out of range");
    #endif
    return mesh().element(neighbourvec[i]);
}

inline
int Element::node_id(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= nodes())
        throw OutOfRangeException("Element::node_id(int): out of range");
    #endif
    return nodevec[i];
}

inline
int Element::edge_id(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= nodes())
        throw OutOfRangeException("Element::edge_id(int): out of range");
    #endif
    return edgevec[i];
}

inline
int Element::face_id(int i) const {
    #ifdef MESH_DEBUG
    if (i < 0 || i >= nodes())
        throw OutOfRangeException("Element::face_id(int): out of range");
    #endif
    return facevec[i];
}

inline
Point Element::centroid() const {
    return my_centroid;
}

inline
void Element::add_scv(int id) {
    scvvec.push_back(id);
}

inline
void Element::add_cvface(int id) {
    cvfacevec.push_back(id);
}

inline
void Element::add_neighbour(int id) {
    neighbourvec.push_back(id);
}

inline
Element::Element() : m(0), my_id(-1), my_centroid() {}

inline
Element::Element(const Mesh& mesh, int type, int id,
                 const std::vector<int>& nodevec,
                 const std::vector<int>& edgevec,
                 const std::vector<int>& facevec,
                 int physical_tag)
    : m(&mesh), my_id(id), my_physical_tag(physical_tag), my_type(type),
      nodevec(nodevec), edgevec(edgevec), facevec(facevec)
{
    for (int i = 0; i < nodes(); ++i)
        my_centroid += node(i).point();
    my_centroid /= nodes();
}

inline
std::ostream &operator<<(std::ostream &cout, Element el)
{
    cout << "Element " << el.id();
    switch( el.type() )
    {
        case 2 : // triangle
            cout << "\t triangle" << std::endl;
            break;
        case 3 : // quad
            cout << "\t quadrilateral" << std::endl;
            break;
        case 4 : // tet
            cout << "\t tetrahadron" << std::endl;
            break;
        case 5 : // hexahedron/brick
            cout << "\t hexahedron" << std::endl;
            break;
        case 6 : // prism
            cout << "\t prism" << std::endl;
            break;
        default :
            cout << "\t unknown element : " << el.type() << std::endl;
    }
    cout << "\tNodes : ";
    for( int i=0; i<el.nodes(); i++ ){
        cout << el.node_id(i) << " ";
    }
    cout << "\tcoordinates\t";
    for( int i=0; i<el.nodes(); i++ ){
        cout << el.mesh().node(el.node_id(i)).point() << "  ";
    }
    cout << std::endl;
    cout << "\tedges : ";
    for( int i=0; i<el.edges(); i++ ){
        cout << el.edge_id(i) << " ";
    }
    cout << "\tmidpoints:\t";
    for( int i=0; i<el.edges(); i++ ){
        cout << el.mesh().edge(el.edge_id(i)).midpoint() << " ";
    }
    cout << std::endl;
    cout << "\tfaces : ";
    for( int i=0; i<el.faces(); i++ ){
        cout << el.face_id(i) << " ";
    }
    cout << std::endl;
    return cout;
}

} // end namespace mesh

#endif
