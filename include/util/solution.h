#ifndef SOLUTION_H
#define SOLUTION_H

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include <fvm/fvm.h>
#include <fvm/physics_base.h>
#include <util/doublevector.h>
#include <fvm/mesh.h>
#include <util/streamstring.h>

#include <lin/lin.h>

namespace util {

template<typename T>
struct block_traits {
    enum {blocksize = T::variables};
};
template<>
struct block_traits<double> {
    enum {blocksize = 1};
};

template<typename TypeName>
class Solution {
public:
    Solution(mpi::MPICommPtr comm): mpicomm_(comm->duplicate("solution")) { };
    typedef lin::Vector<double, lin::DefaultCoordinator<double> > TVec;

    int solutions( void ) const;
    void add( double time, TVec &sol);
    void write_to_file( std::string file_name ) const;
    void write_timestep_VTK( int ts, const mesh::Mesh &m, std::string file_name ) const;
    void write_timestep_VTK_XML( int ts, const mesh::Mesh &m, std::string file_name ) const;
    void write_timestep_header_VTK_XML( int ts, const mesh::Mesh &m, std::string file_name ) const;
private:
    mpi::MPICommPtr mpicomm_;
    std::vector<double> tvec;
    std::vector<TVec> solvec;
};

template<typename TypeName>
int Solution<TypeName>::solutions( void ) const{
    return tvec.size();
}

template<typename TypeName>
void Solution<TypeName>::add( double time, TVec &sol ){
    *mpicomm_ << "saving solution at time " << time << std::endl;
    tvec.push_back(time);
    solvec.push_back( sol );
}

template<typename TypeName>
void Solution<TypeName>::write_to_file( std::string file_name ) const{
    int blocksize = block_traits<TypeName>::blocksize;

    // open the file for input
    std::ofstream fid;
    fid.open(file_name.c_str());
    assert( fid );

    // write the header
    fid << solutions() << " " << blocksize;

    // only write more if there are timesteps to write
    if( solutions() ){
        int len = solvec[0].size();

        // write the number of nodes
        fid << " " << len << std::endl;

        fid << std::scientific << std::setprecision(15);
        // write each timestep
        for( int i=0; i<solutions(); i++ ){
            // write the solution variables at node i
            for( int j=0; j<len; j++ )
                fid << solvec[i][j] << std::endl;
        }
    }

    // close the file
    fid.close();
}

/*
 * output the solution at a given timestep in the vtk file format
 *
 * PRE:
 *      ts  : the timestep index to output
 *      m   : the mesh on which the solution is defined
 *
 */
template<typename TypeName>
void Solution<TypeName>::write_timestep_VTK( int ts, const mesh::Mesh &m, std::string file_name ) const{

    int blocksize = block_traits<TypeName>::blocksize;

    // assert that the user has asked for a valid timestep to output
    assert(ts>=0 && ts<solutions());

    // map from FVMPor element types to VTK cell types
    std::map<int,int> VTK_element_types;
    VTK_element_types[2] = 5;  // triangle
    VTK_element_types[3] = 9;  // quadrilateral
    VTK_element_types[4] = 10; // tet
    VTK_element_types[5] = 12; // hexahedron

    // open file for output
    std::ofstream fid;
    fid.open(file_name.c_str());
    assert( fid );

    // write vtk file header
    fid << "# vtk DataFile Version 2.0" << std::endl << "FVMPor Simulation Output" << std::endl << "ASCII" << std::endl;
    fid << "DATASET UNSTRUCTURED_GRID" << std::endl;

    // write the node positions
    fid << std::endl;
    fid << "POINTS " << m.nodes() <<  " float" << std::endl;
    for( int i=0; i<m.nodes(); i++ ){
        const mesh::Point& p = m.node(i).point();
        fid << p.x << " " << p.y << " " << p.z << std::endl;
    }
    fid << std::endl;

    // write the elements (cells in vtk's idiom)
    int point_list_length=m.elements(); 
    for( int i=0; i<m.elements(); i++ ){
        point_list_length += m.element(i).nodes();
    }
    fid << "CELLS " << m.elements() << " " << point_list_length << std::endl;
    for(int i=0; i<m.elements(); i++){
        const mesh::Element& el = m.element(i);
        fid << el.nodes();
        for( int j=0; j<el.nodes(); j++ ){
            fid << " " << el.node_id(j);
        }
        fid << std::endl;
    }
    fid << std::endl;

    // write the element
    fid << "CELL_TYPES " << m.elements() << std::endl;
    for(int i=0; i<m.elements(); i++)
        fid << VTK_element_types[ m.element(i).type() ] << std::endl;
    fid << std::endl;

    // write the head data 
    fid << "POINT_DATA " << m.nodes() << std::endl;
    fid << "SCALARS head float 1" << std::endl;
    fid << "LOOKUP_TABLE default" << std::endl;
    const double *data = reinterpret_cast<const double*>(solvec[ts].data());
    for(int i=0; i<m.nodes(); i++)
        fid << data[i*blocksize] << " ";

    fid << "SCALARS true_head float 1" << std::endl;
    fid << "LOOKUP_TABLE default" << std::endl;
    for(int i=0; i<m.nodes(); i++)
        fid << data[i*blocksize]+m.node(i).point().y << " ";

    fid.close();
}

/*
 * output the solution at a given timestep in the vtk file format
 *
 * PRE:
 *      ts  : the timestep index to output
 *      m   : the mesh on which the solution is defined
 *
 */
template<typename TypeName>
void Solution<TypeName>::write_timestep_VTK_XML( int ts, const mesh::Mesh &m, std::string file_name ) const{

    *mpicomm_ << "writing solution " << ts << "in VTK XML format" << std::endl;
    int blocksize = block_traits<TypeName>::blocksize;

    if( m.mpicomm()->rank()==0 ){
        write_timestep_header_VTK_XML( ts, m, file_name );
    }
    std::string this_file_name = file_name + "_dom" + util::to_string(m.mpicomm()->rank()) + "_" + util::to_string(ts) + ".vtu";

    // assert that the user has asked for a valid timestep to output
    assert(ts>=0 && ts<solutions());

    // map from FVMPor element types to VTK cell types
    std::map<int,int> VTK_element_types;
    VTK_element_types[2] = 5;  // triangle
    VTK_element_types[3] = 9;  // quadrilateral
    VTK_element_types[4] = 10; // tet
    VTK_element_types[5] = 12; // hexahedron

    // open file for output
    std::ofstream fid;
    fid.open(this_file_name.c_str());
    assert( fid );

    //int numNodes = m.local_nodes();
    //int numElements = m.local_elements();
    int numNodes = m.nodes();
    int numElements = m.elements();

    //////////////////////////////////////////////////////////////
    // write vtk file header
    //////////////////////////////////////////////////////////////
    fid << "<?xml version=\"1.0\"?>" << std::endl;
    fid << "<VTKFile type=\"UnstructuredGrid\">" << std::endl;
    fid << "  <UnstructuredGrid>" << std::endl;

    //////////////////////////////////////////////////////////////
    // define the local piece of information
    //////////////////////////////////////////////////////////////
    fid << "    <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\"" << numElements << "\">" << std::endl;

    //////////////////////////////////////////////////////////////
    // the computed solution data defined at points
    //////////////////////////////////////////////////////////////
    fid << "      <PointData Scalars=\"" <<  TypeName::var_name(0) << "\">" << std::endl;
    const double *data = reinterpret_cast<const double*>(solvec[ts].data());
    for( int var=0; var<blocksize; var++ ){
        // pressure head
        fid << "        <DataArray Name=\"" << TypeName::var_name(var) << "\" type=\"Float32\" format=\"ascii\">" << std::endl;
        fid << "          ";
        for(int i=0; i<numNodes; i++)
            fid << data[i*blocksize + var] << " ";
        fid << std::endl;
        fid << "        </DataArray>" << std::endl;
    }
    fid << "      </PointData>" << std::endl;
    //////////////////////////////////////////////////////////////
    // the subdomain info is stored in each element
    //////////////////////////////////////////////////////////////
    fid << "      <CellData Scalars=\"dom\">" << std::endl;
    fid << "        <DataArray Name=\"dom\" type=\"Int32\" format=\"ascii\">" << std::endl;
    for(int i=0; i<numElements; i++)
        fid << (i>=m.local_elements() ? -1 : m.mpicomm()->rank()) << " ";
    fid << std::endl;
    fid << "        </DataArray>" << std::endl;
    fid << "      </CellData>" << std::endl;
    //////////////////////////////////////////////////////////////
    // definitions of the points: (x,y,z) coordinates of the nodes
    //////////////////////////////////////////////////////////////
    fid << "      <Points>" << std::endl;
    fid << "        <DataArray NumberOfComponents=\"3\" type=\"Float32\" format=\"ascii\">" << std::endl;
    fid << "          ";
    for( int i=0; i<numNodes; i++ ){
        const mesh::Point& p = m.node(i).point();
        fid << p.x << " " << p.y << " " << p.z << " ";
    }
    fid << std::endl;
    fid << "        </DataArray>" << std::endl;
    fid << "      </Points>" << std::endl;

    //////////////////////////////////////////////////////////////
    // element definitions
    //////////////////////////////////////////////////////////////
    fid << "      <Cells>" << std::endl;
    // connections
    fid << "        <DataArray Name=\"connectivity\" type=\"Int32\" format=\"ascii\">" << std::endl;
    fid << "          ";
    for(int i=0; i<numElements; i++){
        const mesh::Element& el = m.element(i);
        //fid << el.nodes();
        for( int j=0; j<el.nodes(); j++ ){
            fid<< el.node_id(j) << " ";
        }
        fid << " ";
    }
    fid << std::endl;
    fid << "        </DataArray>" << std::endl;
    // offsets
    fid << "        <DataArray Name=\"offsets\" type=\"Int32\" format=\"ascii\">" << std::endl;
    fid << "          ";
    int point_list_length=0; 
    for( int i=0; i<numElements; i++ ){
        point_list_length += m.element(i).nodes();
        fid << point_list_length << " ";
    }
    fid << std::endl;
    fid << "        </DataArray>" << std::endl;
    // element types
    fid << "        <DataArray Name=\"types\" type=\"UInt8\" format=\"ascii\">" << std::endl;
    fid << "          ";
    for(int i=0; i<numElements; i++)
        fid << VTK_element_types[ m.element(i).type() ] << " ";
    fid << std::endl;
    fid << "        </DataArray>" << std::endl;
    fid << "      </Cells>" << std::endl;

    //////////////////////////////////////////////////////////////
    // tail of the file
    //////////////////////////////////////////////////////////////
    fid << "    </Piece>" << std::endl;
    fid << "  </UnstructuredGrid>" << std::endl;
    fid << "</VTKFile>" << std::endl;

    fid.close();
}

/*
 * output the xml header for solution from a a parallel run
 *
 * PRE:
 *      ts  : the timestep index to output
 *      m   : the mesh on which the solution is defined
 *
 */
template<typename TypeName>
void Solution<TypeName>::write_timestep_header_VTK_XML( int ts, const mesh::Mesh &m, std::string file_name ) const{
    *mpicomm_ << "writing solution header " << ts << "in VTK XML format" << std::endl;

    int blocksize = block_traits<TypeName>::blocksize;

    // assert that the user has asked for a valid timestep to output
    assert(ts>=0 && ts<solutions());
    std::string this_file_name = file_name + "_head_" + util::to_string(ts) + ".pvtu";

    // open file for output
    std::ofstream fid;
    fid.open(this_file_name.c_str());
    assert( fid );

    //////////////////////////////////////////////////////////////
    // write vtk file header
    //////////////////////////////////////////////////////////////
    int ghost_level = 0;
    fid << "<?xml version=\"1.0\"?>" << std::endl;
    fid << "<VTKFile type=\"PUnstructuredGrid\">" << std::endl;
    fid << "  <PUnstructuredGrid GhostLevel=\"" << ghost_level << "\">" << std::endl;
    fid << "    <PPointData Scalars=\"" <<  TypeName::var_name(0) << "\">" << std::endl;
    for( int var=0; var<blocksize; var++)
        fid << "      <PDataArray Name=\"" << TypeName::var_name(var) << "\" type=\"Float32\" format=\"ascii\"/>" << std::endl;
    fid << "    </PPointData>" << std::endl;
    fid << "    <PCellData Scalars=\"dom\">" << std::endl;
    fid << "      <PDataArray Name=\"dom\" type=\"Int32\" format=\"ascii\"/>" << std::endl;
    fid << "    </PCellData>" << std::endl;
    fid << "    <PPoints>" << std::endl;
    fid << "      <PDataArray NumberOfComponents=\"3\" type=\"Float32\" format=\"ascii\"/>" << std::endl;
    fid << "    </PPoints>" << std::endl;
    fid << "    <PCells>" << std::endl;
    fid << "      <PDataArray Name=\"connectivity\" type=\"Int32\" format=\"ascii\"/>" << std::endl;
    fid << "      <PDataArray Name=\"offsets\" type=\"Int32\" format=\"ascii\"/>" << std::endl;
    fid << "      <PDataArray Name=\"types\" type=\"UInt8\" format=\"ascii\"/>" << std::endl;
    fid << "    </PCells>" << std::endl;
    for( int i=0; i<m.mpicomm()->size(); i++ ){
        fid << "    <Piece Source=\""<< file_name << "_dom" << i << "_" << ts << ".vtu" << "\"/>" << std::endl;
    }
    fid << "  </PUnstructuredGrid>" << std::endl;
    fid << "</VTKFile>" << std::endl;

    fid.close();
}
}

#endif
