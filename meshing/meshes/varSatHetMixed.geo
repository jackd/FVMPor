width = .5;
height = .5;
lc = height/21 - height*1e-8;
lcBlock = lc/20;
blockDepth = lcBlock;
holeWidth = 0.3*width;
blockWidth = 0.9*width;

// Points in counterclockwise order
Point(1) = {0,          0, 0, lc};
Point(2) = {width,      0, 0, lc};
Point(3) = {width,      height, 0, lc};
Point(4) = {holeWidth,  height, 0, lc};
Point(5) = {0,          height, 0, lc};
Point(6) = {0,          (height-blockDepth)/2, 0, lcBlock};
Point(7) = {0,          (height+blockDepth)/2, 0, lcBlock};
Point(8) = {holeWidth,  (height+blockDepth)/2, 0, lcBlock};
Point(9) = {holeWidth,  (height-blockDepth)/2, 0, lcBlock};

// lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,7};
//Line(6) = {7,8};
//Line(7) = {8,9};
//Line(8) = {9,6};
Line(9) = {6,1};

Line(10) = {7,6};

// extrude the low perm region
extSurf[] = Extrude {blockWidth,0,0} {Line{10}; Layers{width/2/lcBlock}; Recombine;};

Line Loop(30) = {1,2,3,4,5, -extSurf[3],extSurf[0],-extSurf[2] ,9};

Plane Surface(40) = {30};

Physical Surface(100) = {40};
Physical Surface(101) = {extSurf[1]};
Physical Line(1) = {1,2,3,5,10,9};
Physical Line(2) = {4};

Mesh.Color.Triangles={0,0,0};


