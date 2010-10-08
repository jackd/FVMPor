lc = .3;
width = 8;
holeWidth = 0.2;
f1 = 3;
f2 = f1;
f4 = 3;
f3 = 1;

// Points in counterclockwise order
Point(1) = {0, 0, 0, lc/f4};
Point(2) = {8, 0, 0, lc};
Point(3) = {8, 5.6, 0, lc/f3};
Point(4) = {8, 6.1, 0, lc/f3};
Point(5) = {8, 6.5, 0, lc/f3};
Point(6) = {2.25, 6.5, 0, lc/f1};
Point(7) = {0, 6.5, 0, lc/f1};
Point(8) = {0, 6.1, 0, lc/f1};
Point(9) = {0, 5.6, 0, lc/f2};
Point(10) = {1, 4, 0, lc/f2};
Point(11) = {3, 4, 0, lc/f2};
Point(12) = {3, 5, 0, lc/f2};
Point(13) = {1, 5, 0, lc/f2};

Point(14) = {3, 6.1, 0, lc/f1};
Point(15) = {3, 5.6, 0, lc/f1};

// lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,1};
Line(10) = {4,14};
Line(11) = {3,15};

Line(16) = {14,8};
Line(17) = {15,9};

Line(12) = {10,11};
Line(13) = {11,12};
Line(14) = {12,13};
Line(15) = {13,10};

Line Loop(30) = {1,2,11,17,9};
Line Loop(31) = {3,10,16,8,-17,-11};
Line Loop(32) = {4,5,6,7,-16,-10};
Line Loop(33) = {12,13,14,15};

Plane Surface(40) = {30,33};
Plane Surface(41) = {31};
Plane Surface(42) = {32};
Plane Surface(43) = {33};

Physical Surface(100) = {42};
Physical Surface(101) = {41};
Physical Surface(102) = {40};
Physical Surface(103) = {43};

Physical Line(1) = {1,2,3,4,5,7,8,9};
Physical Line(2) = {6};

Mesh.Color.Triangles={0,0,0};
