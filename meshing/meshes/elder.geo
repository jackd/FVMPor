lc = 20 - 1e-8; // small adjustment to ensure correct number of mesh subdivisions
width = 300;
height = 150;
topHeight = 150;

// Points in counterclockwise order
Point(1) = {0, 0,      0, lc};
Point(2) = {width, 0,      0, lc/2};
Point(3) = {width, height,      0, lc/2};
Point(4) = {150, height,      0, lc/2};
Point(5) = {0, height,      0, lc};

// lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,1};

Line Loop(6) = {1, 2, 3, 4, 5};
Plane Surface(7) = {6};

// the sides, which correspond to boundaries in the 2D model
Physical Line(1) = {2,4,5};
Physical Line(2) = {3};
Physical Line(4) = {1};

// assign physical tag to the elements in the volume
Physical Surface(100) = {7};
