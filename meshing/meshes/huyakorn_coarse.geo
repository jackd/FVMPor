Geometry.Surfaces = 1;
Geometry.SurfaceNumbers = 1;
lc = 10.0;
Point(1) = {  0.0,  0.0,   0.0,lc};
Point(2) = {200.0,  0.0,   0.0,lc};
Point(3) = {200.0,  0.0,  20.0,lc};
Point(4) = {  0.0,  0.0,  20.0,lc};
Point(5) = {200.0,  0.0,  50.0,lc};
Point(6) = {  0.0,  0.0,  50.0,lc};
Line(101) = {1,2};
Line(102) = {2,3};
Line(103) = {3,4};
Line(104) = {4,1};
Line(105) = {3,5};
Line(106) = {5,6};
Line(107) = {6,4};
Line Loop(201) = {101,102,103,104};
Line Loop(202) = {-103,105,106,107};
Plane Surface(301) = {201};
Plane Surface(302) = {202};
out1[]=Extrude {0,50.0,0.0}{
	Surface{301};
 };
out2[]=Extrude {0,50,0}{
	Surface{302};
 };
Physical Volume(10) = {out1[1],out2[]};

// try netgen:
// Mesh.Algorithm3D = 4;
Physical Surface(1) = {323};
// BC type 2 - h = given, c=1
Physical Surface(2) = {341};
Physical Surface(3) = {337,315};
// BC type 4 - no flux
Physical Surface(4) = {301,302,324,346,311};
Physical Surface(5) = {345};
