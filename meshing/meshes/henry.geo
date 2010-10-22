width = 200;
height = 100;
lc = height/11;

// Points in counterclockwise order
Point(1) = {0, 0, 0, lc};
Point(2) = {width, 0, 0, lc};
Point(3) = {width, height, 0, lc};
Point(4) = {0, height, 0, lc};

// lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(30) = {1,2,3,4};

Plane Surface(40) = {30};

Physical Surface(100) = {40};
Physical Line(1) = {1,3};
Physical Line(2) = {2};
Physical Line(3) = {4};
