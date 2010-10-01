width = 1;
height = 1;
depth = 1;
lc = height/21;
holeWidth = width/2;
factor = 1/1.5;

// Points in counterclockwise order
Point(1) = {0, 0, -height, lc};
Point(2) = {width, 0, -height, lc};
Point(3) = {width, 0, 0, lc};
Point(4) = {holeWidth, 0, 0, lc};
Point(5) = {0, 0, 0, lc};

Point(6) = {holeWidth, depth/2, 0, lc};
Point(7) = {0, depth/2, 0, lc};

Point(8) =  {0, depth, -height, lc};
Point(9) =  {width, depth, -height, lc};
Point(10) = {width, depth, 0, lc};
Point(11) = {0, depth, 0, lc};

// the front
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,1};
Line Loop(30) = {1,2,3,4,5};
Plane Surface(40) = {30};

// the bottom
Line(6) = {2,9};
Line(7) = {9,8};
Line(8) = {8,1};
Line Loop(31) = {1,6,7,8};
Plane Surface(41) = {31};

// around the top
Line(9) = {3,10};
Line(10) = {10,11};
Line(11) = {11,7};
Line(12) = {7,5};

// edges of top hole
Line(13) = {7,6};
Line(14) = {6,4};

// top surface without hole
Line Loop(32) = {9,10,11,13,14,-3};
Plane Surface(42) = {32};
// the hole on top
Line Loop(33) = {13,14,4,-12};
Plane Surface(43) = {33};

// vertical edges at back
Line(15) = {9,10};
Line(16) = {11,8};

// left hand side
Line Loop(34) = {5,-8,-16,11,12};
Plane Surface(44) = {34};
// right hand side
Line Loop(35) = {2,9,-15,-6};
Plane Surface(45) = {35};
// back
Line Loop(36) = {10,16,-7,15};
Plane Surface(46) = {36};


Physical Surface(100) = {40};
//Physical Line(1) = {1,2,3,5};
Physical Line(1) = {1,3,5};
Physical Line(2) = {4};
Physical Line(3) = {2};
//Physical Line(2) = {4};


Mesh.Color.Triangles={0,0,0};
