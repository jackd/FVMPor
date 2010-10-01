width = 1;
height = 1;
lc = height/201;
holeWidth = .5;
factor = 1/1.5;

// Points in counterclockwise order
Point(1) = {0, -height, 0, lc};
Point(2) = {width, -height, 0, lc};
Point(3) = {width, 0, 0, lc};
Point(4) = {holeWidth, 0, 0, lc};
Point(5) = {0, 0, 0, lc};

// lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,1};

Line Loop(30) = {1,2,3,4,5};

Plane Surface(40) = {30};

Physical Surface(100) = {40};
//Physical Line(1) = {1,2,3,5};
Physical Line(1) = {1,3,5};
Physical Line(2) = {4};
Physical Line(3) = {2};
//Physical Line(2) = {4};


Mesh.Color.Triangles={0,0,0};
