lc = 50;
lc2 = 40;
width = 1650;
height = 535;
vScale = 1;
s = 0.6822;
w = 0.3697;
b = -0.8206*height;

// Points in counterclockwise order
Point(1) = {0, b*vScale, 0, lc};
Point(2) = {width, b*vScale, 0, lc2};
Point(3) = {width, (b+s*height)*vScale, 0, lc2};
Point(4) = {(b+height)*(width/height)*(w-1)/(s-1)+w*width, 0, 0, lc2};
Point(5) = {w*width, (b+height)*vScale, 0, lc};
Point(6) = {0, (b+height)*vScale, 0, lc};

// lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,1};

Line Loop(30) = {1,2,3,4,5,6};

Plane Surface(40) = {30};

Physical Surface(100) = {40};
Physical Line(1) = {1,4,5};
Physical Line(2) = {2,3};
Physical Line(3) = {6};

Mesh.Color.Triangles={0,0,0};
