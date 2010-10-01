lc = 10 - 1e-8; // small adjustment to ensure correct number of mesh subdivisions
width = 200;
height = 50;
topHeight = 50;

// Points in counterclockwise order
Point(1) = {0, 0,      0, lc};
Point(2) = {width, 0,      0, lc};
Point(3) = {width, height,      0, lc};
Point(4) = {0, height,      0, lc};

// lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

// the sides, which correspond to boundaries in the 2D model
Physical Line(1) = {4};
Physical Line(2) = {3};
Physical Line(3) = {2};
Physical Line(4) = {1};

// assign physical tag to the elements in the volume
Physical Surface(100) = {6};
