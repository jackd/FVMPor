# Dictionary definition of different elements
# key=element type, data=[numNodes, numFaces, order, name]
elementDefs = { 1 : (2, 1, 1, "line1"), \
                2 : (3, 3, 1, "triangle1"), \
                3 : (4, 4, 1, "quadrangle1"), \
                4 : (4, 4, 1, "tetrahedron1"), \
                5 : (8, 6, 1, "hexahedron1"), \
                6 : (6, 5, 1, "prism1"), \
                7 : (5, 5, 1, "pyramid1") }

# valid element types
faceTypes2D = [1]
faceTypes3D = [2,3]

# valid element types
elementTypes2D = [2,3]
elementTypes3D = [4,5,6]

# node order of faces
faceNodeOrders2D = {  2 : [[0,1],[1,2],[2,0]],\
                      3 : [[0,1],[1,2],[2,3],[3,0]]};
faceNodeOrders3D = {  4 : [[0,1,2], [0,1,3], [0,2,3], [1,2,3]], \
                      5 : [[1,0,3,2], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]], \
                      6 : [[0,1,2], [3,4,5], [0,1,4,3], [1,2,5,4], [2,0,3,5]] };

def permVec(v,p):
    o = []
    for i in range(len(p)):
        o.append(v[p[i]])
    return o

