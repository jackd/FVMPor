#include <fvm/mesh.h>
#include <util/quadrature3d.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <map>
#include <vector>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

namespace mesh {

struct X {
    double operator()(double x, double y, double z) const {
        return x;
    }
};

struct Y {
    double operator()(double x, double y, double z) const {
        return y;
    }
};

struct Z {
    double operator()(double x, double y, double z) const {
        return z;
    }
};

template<typename T>
std::string to_string(const T& t);
double pyramid_volume(CVFace_shape face, Point apex);
double triangle_volume(Point p1, Point p2, Point p3);
std::pair<int, std::pair<int, int> >
get_back_face_and_edges(const Element& e, const Edge& edge, const Node& node);
std::pair<CVFace_shape, CVFace_shape>
make_back_faces(const Element& e, const Edge& edge);
std::pair<int, int> find_RCM_from_edges( const std::vector<std::pair<int, int> > &edges, std::vector<int> &p );

// Mesh file format:
// n_dom dom_id
// vtxdist
// n_nodes_gbl n_nodes_int n_nodes_bnd n_nodes_ext n_elements_int n_elements_bnd
// global-id (repeat)
// x y z boundary-tag (repeat)
// n_nodes n_edges n_faces [node-ids] [boundary-tags] (repeat)

// basic idea:
//  internal stuff is stuff that belongs to and is referenced by this domain only
//  boundary stuff is stuff that belongs to this domain, but is referenced by another domain
//  external stuff is stuff that belongs to another domain but is referenced by this domain

// n_nodes_gbl_: number of nodes in entire mesh
// n_nodes_int_: number of nodes belonging to and referenced by this domain only
// n_nodes_bnd_: number of nodes belonging to this domain, but referenced by other domain(s)
// n_nodes_ext_: number of nodes belonging to other domain(s), but referenced by this domain
// n_elements_int: number of elements not repeated by any other domain
// n_elements_bnd: number of elements repeated by at least one other domain

Mesh::Mesh(const std::string& meshname, mpi::MPICommPtr comm)
    : n_faces_int(0), n_faces_bnd(0),
      n_cvfaces_int(0), n_cvfaces_bnd(0)
{
    mpicomm_ = comm->duplicate("MESH");
    std::ifstream infile, propfile;
    open_mesh_file(meshname, infile, propfile);
    read_mesh_data(infile, propfile);
    construct_control_volumes();
    construct_node_pattern();
}

void Mesh::open_mesh_file(const std::string& meshname,
                          std::ifstream& infile,
                          std::ifstream& propfile) {

    std::string filename = meshname + "_" +
                           to_string(mpicomm_->size()) + "_" +
                           to_string(mpicomm_->rank()) + ".pmesh";

    infile.open(filename.c_str());
    if (!infile)
        throw IOException("Couldn't open file: " + filename);

// each element has a property index assigned to it.
// The physical properties associated with that index are then specified in one global
// property file .prop
    filename = meshname + ".prop";

    propfile.open(filename.c_str());
}

void Mesh::read_mesh_data(std::ifstream& infile, std::ifstream& propfile) {
    int n_nodes_ext, n_nodes, n_elements;
    read_header_data(infile, n_nodes_ext, n_nodes, n_elements);
    //read_properties(propfile, n_elements);
    read_external_nodes(infile, n_nodes_ext);
    read_nodes(infile, n_nodes);
    std::set<Edge> edgeset;
    std::set<Face> faceset;
    read_elements(infile, n_elements, edgeset, faceset);
    //reorder_nodes_edges(edgeset, faceset);
    populate_edges_and_faces(edgeset, faceset);
    set_element_neighbours();
    face_edge_sanity_check();
    element_cvface_sanity_check();
    boundary_tag_sanity_check();
}

void Mesh::reorder_nodes_edges(std::set<Edge> &edgeset, std::set<Face> &faceset){
    /*******************************************
     * Find a node reordering that minimises
     * bandwidth
     *******************************************/
    // generate the edgeset pairs
    int Nedges = edgeset.size();
    std::vector<std::pair<int, int> > edges;

    for(std::set<Edge>::const_iterator it = edgeset.begin(); it != edgeset.end(); ++it)
        edges.push_back(std::make_pair((*it).front().id(), (*it).back().id()));
    for(int i=0; i<nodes(); i++)
        edges.push_back(std::make_pair(i, i));

    // allocate memory for the permutation vector
    std::vector<int> p(nodes());

    // find the RCM ordering
    std::pair<int, int> bw = find_RCM_from_edges( edges, p );

    /*******************************************
     * relabel the nodes and any references to
     * them
     *******************************************/
    // update node ids
    for(int i=0; i<nodes(); i++)
        nodevec[p[i]].my_id = i;
    std::sort(nodevec.begin(), nodevec.end());

    // update element node references
    std::vector<int> q(nodes());
    for(int i=0; i<nodes(); i++)
        q[p[i]] = i;
    for(int i=0; i<elements(); i++)
        for(int j=0; j<elementvec[i].nodes(); j++)
            elementvec[i].nodevec[j] = q[elementvec[i].nodevec[j]];

    /*******************************************
     * sort edges to reduce bandwidth
     *******************************************/
    // make a list of the edge - index pairs
    std::vector<std::pair<Edge,int> > edgesort;
    std::set<Edge>::const_iterator edge_it = edgeset.begin();
    for( int i=0; edge_it!=edgeset.end(); ++i, ++edge_it)
        edgesort.push_back(std::make_pair(*edge_it,edge_it->id()));

    // update the edge information
    for( int i=0; i<edgesort.size(); ++i){
        edgesort[i].first.front_id = q[edgesort[i].first.front_id];
        edgesort[i].first.back_id  = q[edgesort[i].first.back_id];
    }
    // sort the edges by node id
    std::sort(edgesort.begin(), edgesort.end());

    // find the inverse permutation for the edges
    std::vector<int> qedges(edgesort.size());
    for(int i=0; i<qedges.size(); i++)
        qedges[edgesort[i].second] = i;

    // update the edge information
    for( int i=0; i<edgesort.size(); ++i)
        edgesort[i].first.my_id = i;

    // update the element edge references
    for(int i=0; i<elements(); i++)
        for(int j=0; j<elementvec[i].edges(); j++){
            elementvec[i].edgevec[j] = qedges[elementvec[i].edgevec[j]];
        }

    // store the sorted edges in edgevec
    edgevec.resize(edgeset.size());
    for( int i=0; i<edgesort.size(); i++)
        edgevec[i] = edgesort[i].first;

    // copy over faces in id order
    facevec.resize(faceset.size());
    for (std::set<Face>::const_iterator it = faceset.begin(); it != faceset.end(); ++it) {
        facevec[it->id()] = *it;
        if (it->boundary() == 0)
            ++n_faces_int;
        else
            ++n_faces_bnd;
    }

    // update the node and edge references for each face
    for(int i=0; i<facevec.size(); i++){
        for(int j=0; j<facevec[i].nodevec.size(); j++)
            facevec[i].nodevec[j] = q[facevec[i].nodevec[j]];
        for(int j=0; j<facevec[i].edgevec.size(); j++)
            facevec[i].edgevec[j] = qedges[facevec[i].edgevec[j]];
    }
}


void Mesh::read_properties(std::ifstream& propfile, int n_elements) {
    if (!propfile || n_elements == 0) return;

    for (int i = 0; i < n_physical_props; ++i) {
        std::string line;
        std::getline(propfile, line);
        std::istringstream iss(line);
        std::istream_iterator<double> it(iss), end;
        properties[i].assign(it, end);
    }
    if (!propfile)
        throw IOException("Invalid property file");

    std::size_t prop_count = properties[0].size();
    for (int i = 1; i < n_physical_props; ++i) {
        if (properties[i].size() != prop_count) {
            throw IOException("Invalid data in property file");
        }
    }
}

void Mesh::read_header_data(std::ifstream& infile,
    int& n_nodes_ext, int& n_nodes, int& n_elements) {

    // read domain info
    infile >> n_dom >> dom_id;
    if (!infile)
        throw IOException("Couldn't read domain info in file");

    // read vtxdist
    vtx_dist.reserve(n_dom);
    for (int i = 0; i <= n_dom; ++i) {
        int vdist = 0;
        infile >> vdist;
        vtx_dist.push_back(vdist);
    }
    if (!infile)
        throw IOException("Couldn't read vtxdist info");

    // read node count info
    infile >> n_nodes_gbl_ >> n_nodes_int_ >> n_nodes_bnd_ >> n_nodes_ext_;

    if (!infile)
        throw IOException("Couldn't read node count info");
    n_nodes_loc_ = n_nodes_int_ + n_nodes_bnd_;

    // read element count info
    infile >> n_elements_int >> n_elements_bnd;

    if (!infile)
        throw IOException("Couldn't read element count info");
    n_elements = n_elements_int + n_elements_bnd;

    // return values (do we actuall need to set these?)
    n_nodes_ext = n_nodes_ext_;
    n_nodes = n_nodes_loc_ + n_nodes_ext_;
}

void Mesh::read_external_nodes(std::ifstream& infile, int n_nodes_ext) {
    nodes_ext.reserve(n_nodes_ext);
    for (int i = 0; i < n_nodes_ext; ++i) {
        int ext = 0;
        infile >> ext;
        nodes_ext.push_back(ext);
    }
    if (!infile)
        throw IOException("Couldn't read external node ids");
}

void Mesh::read_nodes(std::ifstream& infile, int n_nodes) {
    nodevec.reserve(n_nodes);
    for (int id = 0; id < n_nodes; ++id) {
        int nbc, bc;
        std::vector<int> bcs;
        Point p;
        infile >> p.x >> p.y >> p.z;
        infile >> nbc;
        for( int i=0; i<nbc; i++ ){
            infile >> bc;
            bcs.push_back( bc );
        }
        nodevec.push_back(Node(*this, id, bcs, p));
        if (!infile)
            throw IOException("Couldn't read nodes");
    }
}

void Mesh::read_elements(
    std::ifstream& infile, int n_elements,
    std::set<Edge>& edgeset, std::set<Face>& faceset)
{
    elementvec.reserve(n_elements);
    mesh_dim_ = 3;
    for (int element_id = 0; element_id < n_elements; ++element_id) {

        // read number of nodes, edges, faces
        int n_nodes, n_edges, n_faces, physical_tag, type;
        infile >> type >> physical_tag;
        physical_tag -= 100;
        switch( type ) {
            case 2: // triangle
                n_nodes = 3;
                n_edges = 3;
                n_faces = 3;
                mesh_dim_ = 2;
                break;
            case 3: // quadrilateral
                n_nodes = 4;
                n_edges = 4;
                n_faces = 4;
                mesh_dim_ = 2;
                break;
            case 4: // tetrahedron
                n_nodes = 4;
                n_edges = 6;
                n_faces = 4;
                break;
            case 5: // hexahedron
                n_nodes = 8;
                n_edges = 12;
                n_faces = 6;
                break;
            case 6: // prism
                n_nodes = 6;
                n_edges = 9;
                n_faces = 5;
                break;
            default :
                std::cout << "element type " << type << std::endl;
                throw IOException("ERROR : invalid element type in .pmesh file");
        }

        if (!infile)
            throw IOException("Couldn't read elements");

        // read element's node ids
        std::vector<int> node_ids;
        node_ids.reserve(n_nodes);
        for (int node_id = 0; node_id < n_nodes; ++node_id) {
            int id = 0;
            infile >> id;
            node_ids.push_back(id);
        }
        if (!infile)
            throw IOException("Couldn't read elements");

        // read face boundary ids
        std::vector<int> boundary_ids;
        boundary_ids.reserve(n_faces);
        for (int i = 0; i < n_faces; ++i) {
            int id = 0;
            infile >> id;
            boundary_tags.insert(id);
            boundary_ids.push_back(id);
        }
        if (!infile)
            throw IOException("Couldn't read elements");


        // construct edges and faces
        std::vector<int> edge_ids;
        std::vector<int> face_ids;
        construct_edges_and_faces(
            //n_faces,
            type,
            node_ids, boundary_ids,
            edge_ids, face_ids,
            edgeset, faceset
        );

        // construct element
        elementvec.push_back(
            //Element(*this, element_id, node_ids, edge_ids, face_ids)
            //Element(*this, element_id, node_ids, edge_ids, face_ids, physical_tag)
            Element(*this, type, element_id, node_ids, edge_ids, face_ids, physical_tag)
        );
    }
}

void Mesh::construct_edges_and_faces(
    //int n_faces,
    int type,
    const std::vector<int>& node_ids, const std::vector<int>& boundary_ids,
    std::vector<int>& edge_ids, std::vector<int>& face_ids,
    std::set<Edge>& edgeset, std::set<Face>& faceset)

    // Computes edges and faces for an element.
    // The edges and faces are compared to those already computed for previous
    // elements, and if they're duplicates they aren't added to the set (rather,
    // the id of the existing, equivalent edge/face is used instead).
{
    switch(type) {

    case 2: // triangle
        edge_ids.push_back(insert_edge( edgeset, Edge(*this, edgeset.size(), node_ids[0], node_ids[1]) ));
        edge_ids.push_back(insert_edge( edgeset, Edge(*this, edgeset.size(), node_ids[1], node_ids[2]) ));
        edge_ids.push_back(insert_edge( edgeset, Edge(*this, edgeset.size(), node_ids[2], node_ids[0]) ));
        face_ids.push_back(insert_face( faceset, Face::line( *this, faceset.size(), boundary_ids[0], node_ids[0], node_ids[1], edge_ids[0] )) );
        face_ids.push_back(insert_face( faceset, Face::line( *this, faceset.size(), boundary_ids[1], node_ids[1], node_ids[2], edge_ids[1] )) );
        face_ids.push_back(insert_face( faceset, Face::line( *this, faceset.size(), boundary_ids[2], node_ids[2], node_ids[0], edge_ids[2] )) );
        break;

    case 3: // quadrilateral
        //   id:  0   1   2   3
        // -----------------------------
        // edge: 0-1 1-2 2-3 3-0
        edge_ids.push_back(insert_edge( edgeset, Edge(*this, edgeset.size(), node_ids[0], node_ids[1]) ));
        edge_ids.push_back(insert_edge( edgeset, Edge(*this, edgeset.size(), node_ids[1], node_ids[2]) ));
        edge_ids.push_back(insert_edge( edgeset, Edge(*this, edgeset.size(), node_ids[2], node_ids[3]) ));
        edge_ids.push_back(insert_edge( edgeset, Edge(*this, edgeset.size(), node_ids[3], node_ids[0]) ));
        face_ids.push_back(insert_face( faceset, Face::line( *this, faceset.size(), boundary_ids[0], node_ids[0], node_ids[1], edge_ids[0] )) );
        face_ids.push_back(insert_face( faceset, Face::line( *this, faceset.size(), boundary_ids[1], node_ids[1], node_ids[2], edge_ids[1] )) );
        face_ids.push_back(insert_face( faceset, Face::line( *this, faceset.size(), boundary_ids[2], node_ids[2], node_ids[3], edge_ids[2] )) );
        face_ids.push_back(insert_face( faceset, Face::line( *this, faceset.size(), boundary_ids[3], node_ids[3], node_ids[0], edge_ids[3] )) );
        break;

    case 4: // tetrahedron

        //   id:  0   1   2   3   4   5 
        // -----------------------------
        // edge: 0-1 1-2 2-0 0-3 1-3 2-3

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[0], node_ids[1])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[1], node_ids[2])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[2], node_ids[0])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[0], node_ids[3])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[1], node_ids[3])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[2], node_ids[3])
        ));

        face_ids.push_back(insert_face(
            faceset,
            Face::triangular(
                *this, faceset.size(), boundary_ids[0],
                node_ids[0], node_ids[1], node_ids[2],
                edge_ids[0], edge_ids[1], edge_ids[2]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::triangular(
                *this, faceset.size(), boundary_ids[1],
                node_ids[0], node_ids[1], node_ids[3],
                edge_ids[0], edge_ids[4], edge_ids[3]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::triangular(
                *this, faceset.size(), boundary_ids[2],
                node_ids[0], node_ids[2], node_ids[3],
                edge_ids[2], edge_ids[5], edge_ids[3]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::triangular(
                *this, faceset.size(), boundary_ids[3],
                node_ids[1], node_ids[2], node_ids[3],
                edge_ids[1], edge_ids[5], edge_ids[4]
            ))
        );

        break;

        case 6: // triangular prism

        //   id:  0   1   2   3   4   5   6   7   8 
        // -----------------------------------------
        // edge: 0-1 1-2 2-0 3-4 4-5 5-3 0-3 1-4 2-5

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[0], node_ids[1])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[1], node_ids[2])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[2], node_ids[0])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[3], node_ids[4])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[4], node_ids[5])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[5], node_ids[3])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[0], node_ids[3])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[1], node_ids[4])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[2], node_ids[5])
        ));

        face_ids.push_back(insert_face(
            faceset,
            Face::triangular(
                *this, faceset.size(), boundary_ids[0],
                node_ids[0], node_ids[1], node_ids[2],
                edge_ids[0], edge_ids[1], edge_ids[2]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::triangular(
                *this, faceset.size(), boundary_ids[1],
                node_ids[3], node_ids[4], node_ids[5],
                edge_ids[3], edge_ids[4], edge_ids[5]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::rectangular(
                *this, faceset.size(), boundary_ids[2],
                node_ids[0], node_ids[1], node_ids[4], node_ids[3],
                edge_ids[0], edge_ids[7], edge_ids[3], edge_ids[6]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::rectangular(
                *this, faceset.size(), boundary_ids[3],
                node_ids[1], node_ids[2], node_ids[5], node_ids[4],
                edge_ids[1], edge_ids[8], edge_ids[4], edge_ids[7]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::rectangular(
                *this, faceset.size(), boundary_ids[4],
                node_ids[2], node_ids[0], node_ids[3], node_ids[5],
                edge_ids[2], edge_ids[6], edge_ids[5], edge_ids[8]
            ))
        );

        break;

        case 5: // hexahedron

        //   id:  0   1   2   3   4   5   6   7   8   9   10  11
        // -----------------------------------------------------
        // edge: 0-1 1-2 2-3 3-0 4-5 5-6 6-7 7-4 0-4 1-5 2-6 3-7

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[0], node_ids[1])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[1], node_ids[2])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[2], node_ids[3])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[3], node_ids[0])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[4], node_ids[5])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[5], node_ids[6])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[6], node_ids[7])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[7], node_ids[4])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[0], node_ids[4])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[1], node_ids[5])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[2], node_ids[6])
        ));

        edge_ids.push_back(insert_edge(
            edgeset,
            Edge(*this, edgeset.size(), node_ids[3], node_ids[7])
        ));

        face_ids.push_back(insert_face(
            faceset,
            Face::rectangular(
                *this, faceset.size(), boundary_ids[0],
                node_ids[1], node_ids[0], node_ids[3], node_ids[2],
                edge_ids[0], edge_ids[3], edge_ids[2], edge_ids[1]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::rectangular(
                *this, faceset.size(), boundary_ids[1],
                node_ids[4], node_ids[5], node_ids[6], node_ids[7],
                edge_ids[4], edge_ids[5], edge_ids[6], edge_ids[7]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::rectangular(
                *this, faceset.size(), boundary_ids[2],
                node_ids[0], node_ids[1], node_ids[5], node_ids[4],
                edge_ids[0], edge_ids[9], edge_ids[4], edge_ids[8]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::rectangular(
                *this, faceset.size(), boundary_ids[3],
                node_ids[1], node_ids[2], node_ids[6], node_ids[5],
                edge_ids[1], edge_ids[10],edge_ids[5], edge_ids[9]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::rectangular(
                *this, faceset.size(), boundary_ids[4],
                node_ids[2], node_ids[3], node_ids[7], node_ids[6],
                edge_ids[2], edge_ids[11],edge_ids[6], edge_ids[10]
            ))
        );

        face_ids.push_back(insert_face(
            faceset,
            Face::rectangular(
                *this, faceset.size(), boundary_ids[5],
                node_ids[3], node_ids[0], node_ids[4], node_ids[7],
                edge_ids[3], edge_ids[8], edge_ids[7], edge_ids[11]
            ))
        );

        break;

    default:
        throw IOException("Unsupported element");
    }
}

void Mesh::populate_edges_and_faces(
    const std::set<Edge>& edgeset, const std::set<Face>& faceset)
{
    // copy over edges in id order
    edgevec.resize(edgeset.size());
    for (std::set<Edge>::const_iterator it = edgeset.begin(); it != edgeset.end(); ++it) {
        edgevec[it->id()] = *it;
    }

    // copy over faces in id order
    facevec.resize(faceset.size());
    for (std::set<Face>::const_iterator it = faceset.begin();
         it != faceset.end(); ++it) {
        facevec[it->id()] = *it;
        if (it->boundary() == 0)
            ++n_faces_int;
        else
            ++n_faces_bnd;
    }
}

void Mesh::set_element_neighbours() {

    std::multimap<int,int> elements_with_same_face;

    // record which elements share each face
    for (int element_id = 0; element_id < elements(); ++element_id) {
        const Element& e = element(element_id);
        for (int i = 0; i < e.faces(); ++i) {
            const Face& f = e.face(i);
            elements_with_same_face.insert(std::make_pair(f.id(), e.id()));
        }
    }

    // update the elements to record their neighbours
    typedef std::multimap<int,int>::const_iterator iterator;
    iterator it = elements_with_same_face.begin();
    iterator end = elements_with_same_face.end();

    while(it != end) {
        iterator next_it = it; ++next_it;
        if (next_it == end) break;
        if (it->first == next_it->first) {
            elementvec[it->second].add_neighbour(next_it->second);
            elementvec[next_it->second].add_neighbour(it->second);
            ++next_it;
        }
        it = next_it;
    }
}

void Mesh::face_edge_sanity_check() {
    // Check that the edges connect the face nodes in order
    // (although the direction of the edges is unspecified)
    for (int fid = 0; fid < faces(); ++fid) {
        const Face& f = face(fid);
        for (int i = 0; i < f.edges(); ++i) {
            #ifndef NDEBUG
            int n1 = f.node(i).id();
            int n2 = f.node((i+1)%f.nodes()).id();
            #endif
            assert((f.edge(i).front().id() == n1 && f.edge(i).back().id() == n2) ||
                   (f.edge(i).front().id() == n2 && f.edge(i).back().id() == n1));
        }
    }
}

void Mesh::element_cvface_sanity_check() {
    for (int i = 0; i < elements(); ++i) {
        const Element& e = element(i);
        for (int j = 0; j < e.cvfaces(); ++j)
            assert(e.cvface(j).element().id() == i);
    }
}

void Mesh::boundary_tag_sanity_check() {
    // Check that boundary tag zero is present
    std::set<int>::iterator it = boundary_tags.find(0);
    assert(it != boundary_tags.end());
}

void Mesh::construct_control_volumes() {
    initialise_volumes_and_faces();
    if( dim()==3 ){
        construct_scv_faces_internal_3D();
        construct_scv_faces_boundary_3D();
    }
    else{
        construct_scv_faces_internal_2D();
        construct_scv_faces_boundary_2D();
    }
    /**************************
     * reorganise the CV faces
     * to minimise bandwidth
     **************************/
    // sort the CV faces
    // this sorts by edge id then element id
    std::sort(cvfacevec.begin(), cvfacevec.begin()+interior_cvfaces());
    std::vector<int> q(cvfaces());
    for(int i=0; i<interior_cvfaces(); i++){
        q[cvfacevec[i].my_id] = i;
        cvfacevec[i].my_id = i;
    }
    for(int i=interior_cvfaces(); i<cvfaces(); i++)
        q[i] = i; // the boundary CV faces are not reordered

    // update the SCV and element references to CV faces
    for(int i=0; i<elements(); i++)
        for(int j=0; j<elementvec[i].cvfacevec.size(); j++)
            elementvec[i].cvfacevec[j] = q[elementvec[i].cvfacevec[j]];
    for(int i=0; i<scvs(); i++)
        for(int j=0; j<scvvec[i].cvfacevec.size(); j++)
            scvvec[i].cvfacevec[j] = q[scvvec[i].cvfacevec[j]];

    for(int i=0; i<edge_cvfaces.size(); i++)
        for(int j=0; j<edge_cvfaces[i].size(); j++)
            edge_cvfaces[i][j] = q[edge_cvfaces[i][j]];

    // finish of the CVs
    construct_volumes();
}

void Mesh::initialise_volumes_and_faces() {
    // Create empty volumes for each node
    for (int i = 0; i < nodes(); ++i) {
        volumevec.push_back(Volume(*this, i));
    }
    // Create empty SCVs for each node of each element
    // Record the ids in elements and volumes
    for (int i = 0; i < elements(); ++i) {
        Element& e = elementvec[i]; // non-const
        for (int j = 0; j < e.nodes(); ++j) {
            e.add_scv(scvvec.size());
            volumevec[e.node(j).id()].add_scv(scvvec.size());
            scvvec.push_back(SCV(*this, scvvec.size(), i, e.node(j).id()));
        }
    }
}

void Mesh::construct_scv_faces_internal_3D() {

    // Constructs non-boundary SCV faces, and also updates the volumes of
    // the corresponding SCVs.
    edge_cvfaces.resize(edges());

    // for each element
    for (int i = 0; i < elements(); ++i) {
        Element& e = elementvec[i]; // non-const

        // for each of its edges
        // (the CV face will use the edge midpoint as a vertex, and will be the
        // face separating the SCVS for the two nodes of the edge)
        for (int j = 0; j < e.edges(); ++j) {
            const Edge& edge = e.edge(j);

            // find the two element faces that share this edge
            // (the CV face will use both face centroids as vertices)
            std::pair<int, int> facepair = e.faces_from_edge(edge);
            const Face& face1 = e.face(facepair.first);
            const Face& face2 = e.face(facepair.second);

            // add the CV face id to the element
            e.add_cvface(cvfacevec.size());

            // add the CV face id to the SCVs of both nodes
            int front_id = e.node_id(edge.front());
            int back_id = e.node_id(edge.back());
            scvvec[e.scv(front_id).id()].add_cvface(cvfacevec.size());
            scvvec[e.scv(back_id).id()].add_cvface(cvfacevec.size());

            // create the CV face quadrilateral
            CVFace_shape q(
                edge.midpoint(), face1.centroid(), e.centroid(), face2.centroid()
            );

            // Create its two "back" face quadrilaterals.
            // These are the faces that lie on the element face itself -
            // they don't need to be stored (unless they're boundary faces, see
            // later function), but we do need them here to update the volume.
            std::pair<CVFace_shape, CVFace_shape> back_faces
                = make_back_faces(e, edge);

            // Here we're updating the volumes of the two SCVs.  The strategy is
            // to pick an arbitrary point in the middle of the SCV (we use the
            // midpoint of the diagonal joining the node and element centroid),
            // and to compute the volumes of the pyramids made up of the SCV
            // faces connected to the aribitrary point.

            // update the volume of the first SCV
            Point p1 = (e.centroid() + edge.front().point()) / 2.0;
            scvvec[e.scv(front_id).id()].add_volume(
                pyramid_volume(q, p1));
            scvvec[e.scv(front_id).id()].add_volume(
                pyramid_volume(back_faces.first, p1));

            // update the volume of the second SCV
            Point p2 = (e.centroid() + edge.back().point()) / 2.0;
            scvvec[e.scv(back_id).id()].add_volume(
                pyramid_volume(q, p2));
            scvvec[e.scv(back_id).id()].add_volume(
                pyramid_volume(back_faces.second, p2));

            // Here we're recording the eight points that comprise the vertices
            // of the two SCVs.  This was not part of the original
            // specification, and was only added as an afterthought.  Thus it is
            // not particularly elegant - but it works.
            const CVFace_shape& b1 = back_faces.first;
            scvvec[e.scv(front_id).id()].set_vertices3D(
                 q.point(0),  q.point(1),  q.point(2),  q.point(3),
                b1.point(0), b1.point(1), b1.point(2), b1.point(3)
            );
            const CVFace_shape& b2 = back_faces.second;
            scvvec[e.scv(back_id).id()].set_vertices3D(
                 q.point(0),  q.point(1),  q.point(2),  q.point(3),
                b2.point(0), b2.point(1), b2.point(2), b2.point(3)
            );

            // create the CV face itself
            CVFace cvface(
                *this, cvfacevec.size(), i,
                edge.front().id(), edge.back().id(), 0, q, j, edge.id()
            );

            // add CV face to mesh vector
            cvfacevec.push_back(cvface);

            // add the new cv face to the list of faces attached to the edge
            edge_cvfaces[edge.id()].push_back(cvfacevec.size()-1);

            ++n_cvfaces_int;

        } // end for each of its edges

    } // end for each element
}

void Mesh::construct_scv_faces_internal_2D() {

    // Constructs non-boundary SCV faces, and also updates the volumes of
    // the corresponding SCVs.
    edge_cvfaces.resize(edges());

    // for each element
    for (int i = 0; i < elements(); ++i) {
        Element& e = elementvec[i]; // non-const

        // for each of its edges
        // (the CV face will use the edge midpoint and element centroid as vertices
        for (int j = 0; j < e.edges(); ++j) {
            const Edge& edge = e.edge(j);

            // add the CV face id to the element
            e.add_cvface(cvfacevec.size());

            // add the CV face id to the SCVs of both nodes
            int front_id = e.node_id(edge.front());
            int back_id = e.node_id(edge.back());
            scvvec[e.scv(front_id).id()].add_cvface(cvfacevec.size());
            scvvec[e.scv(back_id).id()].add_cvface(cvfacevec.size());

            // create the CV face line (in 2D)
            CVFace_shape f( edge.midpoint(), e.centroid() );

            // Create its two "back" faces.
            // These are the faces that lie on the element face itself -
            // they don't need to be stored (unless they're boundary faces, see
            // later function), but we do need them here to update the volume.
            std::pair<CVFace_shape, CVFace_shape> back_faces
                = make_back_faces(e, edge);

            // Here we're updating the volumes of the two SCVs.  The strategy is
            // to pick an arbitrary point in the middle of the SCV (we use the
            // midpoint of the diagonal joining the node and element centroid),
            // and to compute the volumes of the pyramids made up of the SCV
            // faces connected to the aribitrary point.

            // update the volume of the first SCV
            //scvvec[e.scv(front_id).id()].add_volume( triangle_volume(e.centroid(), edge.midpoint(), node(front_id).point()) );
            scvvec[e.scv(front_id).id()].add_volume( triangle_volume(e.centroid(), edge.midpoint(), node(edge.front().id()).point()) );

            // update the volume of the second SCV
            scvvec[e.scv(back_id).id()].add_volume( triangle_volume(e.centroid(), edge.midpoint(), node(edge.back().id()).point()) );

            // Here we're recording the eight points that comprise the vertices
            // of the two SCVs.  This was not part of the original
            // specification, and was only added as an afterthought.  Thus it is
            // not particularly elegant - but it works.
            const CVFace_shape& b1 = back_faces.first;
            scvvec[e.scv(front_id).id()].set_vertices2D( e.centroid(), edge.midpoint(), b1.point(0), b1.point(1) );
            const CVFace_shape& b2 = back_faces.second;
            scvvec[e.scv(back_id).id()].set_vertices2D( e.centroid(),  edge.midpoint(), b2.point(0), b2.point(1) );

            // create the CV face itself
            CVFace cvface( *this, cvfacevec.size(), i, edge.front().id(), edge.back().id(), 0, f, j, edge.id() );

            // add CV face to mesh vector
            cvfacevec.push_back(cvface);

            // add the new cv face to the list of faces attached to the edge
            edge_cvfaces[edge.id()].push_back(cvfacevec.size()-1);

            ++n_cvfaces_int;
        } // end for each of its edges

    } // end for each element
}

void Mesh::construct_volumes() {
    // Compute the centroids of each SCV.  This was an afterthought.
    if( dim()==3 ){
        for (int i = 0; i < scvs(); ++i) {
            const SCV& s = scv(i);
            util::Quadrature3D I1(
                s.vertex(0), s.vertex(1), s.vertex(2), s.vertex(3),
                s.vertex(4), s.vertex(5), s.vertex(6), s.vertex(7)
            );
            double cx = I1(X(), 3);  // Gaussian quadrature is
            double cy = I1(Y(), 3);  // exact for low-degree
            double cz = I1(Z(), 3);  // polynomials
            scvvec[i].set_centroid(Point(cx, cy, cz) / s.vol());
        }
    }
    else{
        for (int i = 0; i < scvs(); ++i) {
            const SCV& s = scv(i);

            Point centroid1 = (s.vertex(0) + s.vertex(1) + s.vertex(3)) / 3.0;
            Point centroid2 = (s.vertex(1) + s.vertex(2) + s.vertex(3)) / 3.0;
            double area1 = norm(cross(s.vertex(1) - s.vertex(0), s.vertex(3) - s.vertex(0))) / 2.0;
            double area2 = norm(cross(s.vertex(1) - s.vertex(2), s.vertex(3) - s.vertex(2))) / 2.0;
            scvvec[i].set_centroid( (area1*centroid1 + area2*centroid2) / (area1+area2) );
        }
    }

    // Compute the volumes and centroids of each CV
    for (int i = 0; i < nodes(); ++i) {
        volumevec[i].compute_volume();
        volumevec[i].compute_centroid();
    }
}

void Mesh::construct_scv_faces_boundary_2D() {
    // Construct any boundary SCV faces
    // for 2D meshes
    assert( dim()==2 );

    // for each element
    for (int i = 0; i < elements(); ++i) {
        Element& e = elementvec[i]; // non-const

        // for each of its faces
        for (int j = 0; j < e.faces(); ++j) {
            const Face& f = e.face(j);

            // if it's a boundary face
            if (f.boundary()) {

                // for each of its nodes
                for (int k = 0; k < f.nodes(); ++k) {
                    const Node& n = f.node(k);

                    // create CV face quadrilateral
                    CVFace_shape q( n.point(), f.centroid() );

                    // we're responsible for ensuring the orientation is correct
                    // (since it's a boundary CV face)
                    Point outwards = q.centroid() - e.centroid();
                    if (dot(q.normal(), outwards) < 0.0) {
                        q.reverse();
                    }
                    assert(dot(q.normal(), outwards) > 0.0);

                    int id = e.node_id(n);

                    CVFace cvface(  // note: no front face
                        *this, cvfacevec.size(), i, -1, n.id(), f.boundary(), q, -1, -1
                    );

                    // add the CV face id to the element
                    e.add_cvface(cvfacevec.size());

                    // add the CV face id to the SCV
                    scvvec[e.scv(id).id()].add_boundary_cvface(cvfacevec.size());

                    // add CV face to mesh vector
                    cvfacevec.push_back(cvface);

                    ++n_cvfaces_bnd;
                }
            }
        }
    }
}

void Mesh::construct_scv_faces_boundary_3D() {
    // Construct any boundary SCV faces

    // for each element
    for (int i = 0; i < elements(); ++i) {
        Element& e = elementvec[i]; // non-const

        // for each of its faces
        for (int j = 0; j < e.faces(); ++j) {
            const Face& f = e.face(j);

            // if it's a boundary face
            if (f.boundary()) {

                // for each of its nodes
                for (int k = 0; k < f.nodes(); ++k) {
                    const Node& n = f.node(k);

                    // create CV face quadrilateral
                    std::pair<int, int> edges = f.edges_from_node(n);
                    CVFace_shape q(
                        n.point(),
                        f.edge(edges.first).midpoint(),
                        f.centroid(),
                        f.edge(edges.second).midpoint()
                    );

                    // we're responsible for ensuring the orientation is correct
                    // (since it's a boundary CV face)
                    Point outwards = q.centroid() - e.centroid();
                    if (dot(q.normal(), outwards) < 0.0) {
                        q.reverse();
                    }
                    assert(dot(q.normal(), outwards) > 0.0);

                    int id = e.node_id(n);

                    CVFace cvface(  // note: no front face
                        *this, cvfacevec.size(), i, -1, n.id(), f.boundary(), q, -1, -1
                    );

                    // add the CV face id to the element
                    e.add_cvface(cvfacevec.size());

                    // add the CV face id to the SCV
                    scvvec[e.scv(id).id()].add_boundary_cvface(cvfacevec.size());

                    // add CV face to mesh vector
                    cvfacevec.push_back(cvface);

                    ++n_cvfaces_bnd;
                }
            }
        }
    }
}

// generate the Pattern that describes the distribution of nodes
// over the different processors
void Mesh::construct_node_pattern() {
    ///////////////////////////////////////////////////////////////
    // WARNING!
    // this assumes that nodes are shared BOTH ways by adjacent subdomains.
    // That is, if one domain has to send information to a neighbour,
    // that implies that it also has to receive from that neighbour and
    // vice versa.
    ///////////////////////////////////////////////////////////////

    //mpicomm_->log_stream() << "Mesh::construct_node_pattern()" << std::endl << "-----------------------------" << std::endl;
    *mpicomm_ << "Mesh::construct_node_pattern()" << std::endl << "-----------------------------" << std::endl;

    node_pattern_ = Pattern(mpicomm_);

    // make a list of global node ids paired with the local index of each external node for the local domain
    std::vector< std::pair<int, int> > index_vector(external_nodes());
    for (int k = 0; k < external_nodes(); ++k) {
        index_vector[k] = std::make_pair( external_node_id(k), k+n_nodes_loc_);
    }
    // sort the list by global index
    // the list is sorted to ensure that the information will be sent
    // in the order that it is stored on the neighbouring process
    std::sort(index_vector.begin(), index_vector.end());

    // now make a list of nodes to receive from each neighbour
    std::map<int,std::vector<int> > external_node_index_local;
    std::map<int,std::vector<int> > external_node_index_global;
    int numDomains = mpicomm_->size();
    for( int i=0; i<external_nodes(); i++ ){
        int node = index_vector[i].first;
        for( int dom=0; dom<numDomains; dom++){
            if( node<vtx_dist[dom+1] ){
                external_node_index_global[dom].push_back(index_vector[i].first);
                external_node_index_local[dom].push_back(index_vector[i].second);
                break;
            }
        }
    }
    std::vector<int> neighbours;
    for( int i=0; i<numDomains; i++ )
        if( external_node_index_local.count(i) )
            neighbours.push_back(i);

    std::vector<MPI_Request> send_request;
    std::vector<MPI_Request> recv_request;
    std::vector<int> to_send;
    send_request.resize(neighbours.size());
    recv_request.resize(neighbours.size());
    to_send.resize(neighbours.size());

    // send/recv to/from each neighbour
    for( int i=0; i<neighbours.size(); i++ ){
        int n = neighbours[i];
        int size = external_node_index_local[n].size();
        send_request[i] = mpicomm_->Isend( size, n, 1 );
        recv_request[i] = mpicomm_->Irecv( to_send[i], n, 1 );
    }

    // wait for communication to complete
    std::vector<MPI_Status> status;
    status.resize(neighbours.size());
    mpicomm_->Waitall(send_request, status);
    mpicomm_->Waitall(recv_request, status);
    int total_sent = 0;
    int total_received = 0;
    for( int i=0; i<neighbours.size(); i++ ){
        int n = neighbours[i];
        total_sent += to_send[i];
        total_received += external_node_index_local[n].size();
    }
    if(total_received!=n_nodes_ext_)
        throw IOException("Incorrect parallel mesh format : external node data does not match boundary node data on neighbours");
    mpicomm_->log_stream() << "receiving and sending " << total_received << " and " << total_sent << std::endl;

    std::map<int,std::vector<int> > boundary_node_index;
    for( int i=0; i<neighbours.size(); i++ ){
        MPI_Request request;
        int n = neighbours[i];

        boundary_node_index[n].resize(to_send[i]);
        send_request[i] = mpicomm_->Isend( external_node_index_global[n], n, 2 );
        recv_request[i] = mpicomm_->Irecv( boundary_node_index[n], n, 2 );
    }
    mpicomm_->Waitall(send_request, status);
    mpicomm_->Waitall(recv_request, status);

    for( int i=0; i<neighbours.size(); i++ ){
        int n = neighbours[i];
        int startIndex = vtx_dist[mpicomm_->rank()];
        for( int j=0; j<boundary_node_index[n].size(); j++)
            boundary_node_index[n][j] -= startIndex;
        node_pattern_.add_neighbour( n, boundary_node_index[n], external_node_index_local[n] );
    }
    mpicomm_->log_stream() << "Mesh::construct_node_pattern() FINISHED" << std::endl << "-----------------------------" << std::endl;
}

// Attempts to insert an edge into the domain set.  If an equivalent edge exists
// then no insertion is made.  Either way, the id of the edge is returned.
int Mesh::insert_edge(std::set<Edge>& edgeset, const Edge& edge) {
    std::pair<std::set<Edge>::iterator, bool> pr = edgeset.insert(edge);
    return pr.first->id();
}

// Attempts to insert a face into the domain set.  If an equivalent face exists
// then no insertion is made.  Either way, the id of the face is returned.
int Mesh::insert_face(std::set<Face>& faceset, const Face& face) {
    std::pair<std::set<Face>::iterator, bool> pr = faceset.insert(face);
    return pr.first->id();
}

// Returns the index of the edges that share a given node
std::pair<int, int> Face::edges_from_node(const Node& n) const {
    int edge_id1 = -1;
    int edge_id2 = -1;

    // ensure that we are in 3D
    assert( nodes()>2 );
    // count up
    for (int i = 0; i < edges(); ++i) {
        if (edge(i).front().id() == n.id() || edge(i).back().id() == n.id()) {
            edge_id1 = i;
            break;
        }
    }

    // count down
    for (int i = edges()-1; i >= 0; --i) {
        if (edge(i).front().id() == n.id() || edge(i).back().id() == n.id()) {
            edge_id2 = i;
            break;
        }
    }

    assert(edge_id1 != -1);
    assert(edge_id2 != -1);
    assert(edge_id1 != edge_id2);

    return std::make_pair(edge_id1, edge_id2);
}

// Returns the index of node in the element's node list
int Element::node_id(const Node& n) const {
    for (int i = 0; i < nodes(); ++i)
        if (node(i).id() == n.id())
            return i;
    assert(false);  // node not found - bad input to function
    return 0; // suppress warning
}

// Returns the index of edge in the element's edge list
int Element::edge_id(const Edge& e) const {
    for (int i = 0; i < edges(); ++i)
        if (edge(i).id() == e.id())
            return i;
    assert(false);  // edge not found - bad input to function
    return 0; // suppress warning
}

// Returns the index of face in the element's face list
int Element::face_id(const Face& f) const {
    for (int i = 0; i < faces(); ++i)
        if (face(i).id() == f.id())
            return i;
    assert(false);  // face not found - bad input to function
    return 0; // supress warning
}

// Returns the two faces of an element that share the given edge.
std::pair<int, int> Element::faces_from_edge(const Edge& edge) const {

    // ensure that we are only using a 3D element
    assert( dim()==3 );

    // count up
    int face1_id = -1;
    for (int ii = 0; ii < faces(); ++ii) {
        const Face& f = face(ii);
        for (int jj = 0; jj < f.edges(); ++jj) {
            if (f.edge(jj).id() == edge.id()) {
                face1_id = ii;
                ii = faces();   // break out of outer loop
                jj = f.edges(); // break out of inner loop
            }
        }
    }

    // count down
    int face2_id = -1;
    for (int ii = faces()-1; ii >= 0; --ii) {
        const Face& f = face(ii);
        for (int jj = 0; jj < f.edges(); ++jj) {
            if (f.edge(jj).id() == edge.id()) {
                face2_id = ii;
                ii = -1;        // break out of outer loop
                jj = f.edges(); // break out of inner loop
            }
        }
    }

    assert(face1_id != -1);
    assert(face2_id != -1);
    assert(face1_id != face2_id);

    return std::make_pair(face1_id, face2_id);
}

// Returns the face of an element that shares the given edges.
int Element::face_from_edges(const Edge& edge1, const Edge& edge2) const {
    bool found1 = false;
    bool found2 = false;
    int i = 0;

    // ensure that the element is 3D
    assert( dim()==3 );

    while (i < faces() && !(found1 && found2)) {
        found1 = found2 = false;
        const Face& f = face(i);
        int j = 0;
        while (j < f.edges() && !found1) {
            found1 = (f.edge(j).id() == edge1.id() || f.edge(j).id() == edge2.id());
            ++j;
        }
        while (j < f.edges() && !found2) {
            found2 = (f.edge(j).id() == edge1.id() || f.edge(j).id() == edge2.id());
            ++j;
        }
        ++i;
    }
    assert(found1 && found2);   // face not found - bad input to function
    return i-1;
}

// 2D version
int Element::other_edge_for_node( const Node& n, const Edge& edg) const {

    // ensure that this is a 2D element
    assert( dim()==2 );

    // find edge
    int i = 0;
    while(i < edges()) {
        if (edge(i).id() != edg.id() &&
           (edge(i).front().id() == n.id() || edge(i).back().id() == n.id()))
            break;
        ++i;
    }

    // ensure that an edge was found
    assert(i<edges());

    return i;
}
// 3D version
std::pair<int, int> Element::other_edges_for_node(
    const Node& n, const Edge& edg) const {

    // ensure that this is a 3D element
    assert( dim()==3 );

    // find first edge
    int i = 0;
    while(i < edges()) {
        if (edge(i).id() != edg.id() &&
           (edge(i).front().id() == n.id() || edge(i).back().id() == n.id()))
            break;
        ++i;
    }
    // find second edge
    int j = edges()-1;
    while(j >= 0) {
        if (edge(j).id() != edg.id() &&
           (edge(j).front().id() == n.id() || edge(j).back().id() == n.id()))
            break;
        --j;
    }
    assert(i != edges());
    assert(j != -1);
    assert(i != j);
    assert(edge(i).front().id() == n.id() || edge(i).back().id() == n.id());
    assert(edge(j).front().id() == n.id() || edge(j).back().id() == n.id());
    return std::pair<int,int>(i, j);
}

std::pair<int, std::pair<int, int> > get_back_face_and_edges(
    const Element& e, const Edge& edge, const Node& node) {

    std::pair<int, int> other_edges = e.other_edges_for_node(node, edge);
    int face_id = e.face_from_edges(
        e.edge(other_edges.first), e.edge(other_edges.second)
    );
    return std::make_pair(face_id, other_edges);
}

std::pair<CVFace_shape, CVFace_shape>
make_back_faces(const Element& e, const Edge& edge) {

    if( e.dim()==3 ){
        std::pair<int, std::pair<int, int> > face_n_edges;

        face_n_edges = get_back_face_and_edges(e, edge, edge.front());
        CVFace_shape q1(
            edge.front().point(),
            e.edge(face_n_edges.second.first).midpoint(),
            e.face(face_n_edges.first).centroid(),
            e.edge(face_n_edges.second.second).midpoint()
        );

        face_n_edges = get_back_face_and_edges(e, edge, edge.back());
        CVFace_shape q2(
            edge.back().point(),
            e.edge(face_n_edges.second.first).midpoint(),
            e.face(face_n_edges.first).centroid(),
            e.edge(face_n_edges.second.second).midpoint()
            );

        return std::make_pair(q1, q2);
    }
    else{
        int back_edge = e.other_edge_for_node( edge.front(), edge );
        CVFace_shape faceFront( edge.front().point(), e.edge(back_edge).midpoint() );

        back_edge = e.other_edge_for_node( edge.back(), edge );
        CVFace_shape faceBack( edge.back().point(), e.edge(back_edge).midpoint() );

        return std::make_pair(faceFront, faceBack);
    }

}

double triangle_volume(Point p1, Point p2, Point p3) {
    Point v1 = p1-p2;
    Point v2 = p1-p3;

    double a = dot(v1,v1);
    double b = dot(v2,v2);
    double c = dot(v1,v2);
    double d = a*b - c*c;
    assert(d>0);
    return std::sqrt( d ) / 2.;
}

double pyramid_volume(CVFace_shape face, Point apex) {

    // ensure that we are working on a 3D mesh
    assert( face.points()==4 );

    double x = apex.x;
    double y = apex.y;
    double z = apex.z;

    double x1 = face.point(0).x;
    double y1 = face.point(0).y;
    double z1 = face.point(0).z;

    double x2 = face.point(1).x;
    double y2 = face.point(1).y;
    double z2 = face.point(1).z;

    double x3 = face.point(3).x;
    double y3 = face.point(3).y;
    double z3 = face.point(3).z;

    Point normal = face.normal();
    double length = norm(normal);
    double xn = normal.x / length;
    double yn = normal.y / length;
    double zn = normal.z / length;

    double height =
        (
            ( z1*y3+z2*y1-z2*y3+y2*z3-y2*z1-y1*z3)*x+
            (-x2*z3+z2*x3-z1*x3+x2*z1-z2*x1+x1*z3)*y+
            ( y2*x1+x2*y3-x1*y3+y1*x3-y2*x3-x2*y1)*z-
            x2*y3*z1+z2*y3*x1-z2*x3*y1+x2*y1*z3-y2*z3*x1+y2*x3*z1
        ) /
        (-z1*x3*yn-y1*z3*xn+y2*x1*zn+y1*x3*zn+y2*z3*xn+z1*y3*xn-y2*x3*zn-
         y2*z1*xn+z2*y1*xn-z2*y3*xn-z2*x1*yn+z2*x3*yn+x1*yn*z3-x1*y3*zn+
         x2*yn*z1-x2*yn*z3-x2*y1*zn+x2*y3*zn);

    return std::abs(face.area() * height / 3.0);
}

double Mesh::local_vol() const {
    double v = 0.0;
    for (int i = 0; i < local_nodes(); ++i) {
        v += volume(i).vol();
    }
    return v;
}

double Mesh::vol() const {
    double v = 0.0;
    for (int i = 0; i < nodes(); ++i) {
        v += volume(i).vol();
    }
    return v;
}

double Mesh::total_vol() const {
    double vol = local_vol();
    double total_vol = 0.0;
    MPI_Reduce(&vol, &total_vol, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return total_vol;
}

template<typename T>
std::string to_string(const T& t) {
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

std::pair<int, int> find_RCM_from_edges( const std::vector<std::pair<int, int> > &edges, std::vector<int> &p )
{
    using namespace boost;
    typedef adjacency_list< vecS, vecS, undirectedS,
                            property<vertex_color_t, default_color_type,
                            property<vertex_degree_t,int> > > Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    typedef graph_traits<Graph>::vertices_size_type size_type;

    int Nnodes = p.size();
    int Nedges = edges.size();

    // instatiate and populate the graph
    Graph G(Nnodes);
    for(int i=0; i<Nedges; ++i)
        add_edge((size_t)edges[i].first, (size_t)edges[i].second, G);

    graph_traits<Graph>::vertex_iterator ui, ui_end;

    property_map<Graph,vertex_degree_t>::type deg = get(vertex_degree, G);
    for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui)
        deg[*ui] = degree(*ui, G);

    // determine the bandwidth of the original graph
    property_map<Graph, vertex_index_t>::type
    index_map = get(vertex_index, G);
    std::vector<Vertex> inv_perm(num_vertices(G));
    std::cout << "original bandwidth: " << bandwidth(G) << std::endl;
    int bw_init = bandwidth(G);

    //reverse cuthill_mckee_ordering
    cuthill_mckee_ordering(G, inv_perm.rbegin(), get(vertex_color, G),
                           make_degree_map(G));

    int c=0;
    for( std::vector<Vertex>::const_iterator i=inv_perm.begin(); i != inv_perm.end(); ++i ){
        p[c++] = index_map[*i];
    }

    // determine the new bandwidth of the permuted graph
    std::vector<size_type> perm(num_vertices(G));
    for (size_type c = 0; c != inv_perm.size(); ++c)
        perm[index_map[inv_perm[c]]] = c;
    std::cout << "new  bandwidth: " 
              << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]))
              << std::endl;
    int bw_perm = bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]));
    return std::make_pair(bw_init, bw_perm);
}

} // end namespace mesh
