#!/usr/bin/env bash

baseFile=$1
echo
echo -----------------------------------------------------------------------
echo Generating mesh based on $baseFile 
echo -----------------------------------------------------------------------
echo

# test to see if the .geo input file exists
geoFile=${baseFile}.geo
if [ ! -f ${geoFile} ]
then
    echo ERROR : the geometry file ${geoFile} does not exist
    exit
fi

# generate the .mesh file from the .geo file
# only do this if no .mesh file has been created from the .geo file,
# or if the .geo file has been altered since the last .mesh
# file was generated
meshFile=${baseFile}.mesh
if [ ! -f ${meshFile} ] || [ ${meshFile} -ot ${geoFile} ]
then
    # call gmsh
    mshFile=${baseFile}.msh
    logFile=${baseFile}_gmsh.log
    echo \* calling gmsh with ${geoFile} \(see ${logFile} for details\)...
    gmsh ${geoFile} -3 -optimize > ${logFile}
    echo "     finished "$mshFile 

    # call Mesh.py
    meshLog=${baseFile}_mesh.log
    echo \* generating $meshFile from gmsh format file $mshFile \(see ${meshLog} for details\)...
    #python Mesh.py ${baseFile} > ${meshLog}
    python Mesh.py ${baseFile}
    echo "     finished "$meshFile

    # remove the intermediate file
    #rm ${mshFile}
else
    echo The geometry file ${geoFile} has not changed since last time.
    echo There is no need to regenerate ${baseFile}.mesh.
fi

# perform domain decomposition if the user has requested it
if [ $# -eq 2 ]
then
    nProcs=$2
    echo
    echo -----------------------------------------------------------------------
    echo Performing domain decomposition with ${nProcs} domains
    echo -----------------------------------------------------------------------
    echo

    decompLog=${baseFile}_decomp_${nProcs}.log
    splitLog=${baseFile}_split_${nProcs}.log
    echo calling decomp \(see logfile ${decompLog} for details\)...
    #mpirun -np ${nProcs} ../decomp/bin/decomp ${baseFile} > ${decompLog}
    mpirun -np ${nProcs} ../decomp/bin/decomp ${baseFile}
    echo "     finished"
    echo calling split \(see logfile ${splitLog} for details\)...
    #mpirun -np ${nProcs} ../decomp/bin/split ${baseFile} > ${splitLog}
    mpirun -np ${nProcs} ../decomp/bin/split ${baseFile}
    echo "     finished"

    #cleanup intermediate files
    rm ${baseFile}_p_${nProcs}.txt
    rm ${baseFile}_q_${nProcs}.txt
    rm ${baseFile}_${nProcs}.bmesh
    rm ${baseFile}_${nProcs}.perm

    # save the parallel mesh info
    #mv ${baseFile}_${nProcs}.dom ./pmeshes
    #mv ${baseFile}_${nProcs}*.pmesh ./pmeshes
    #mv ${baseFile}*.log  ./logs
fi
