// horizontal scale factor
hs = 1;

// meshing parameter
lc = 3;

Point(1) = {160*hs,-77,0,lc};
Point(2) = {160*hs,4,0,lc};
Point(3) = {101*hs,4,0,lc};
Point(4) = {98*hs,18,0,lc};
Point(5) = {90*hs,25,0,lc};
Point(6) = {60*hs,25,0,lc};
Point(7) = {54*hs,28,0,lc};
Point(8) = {38*hs,33,0,lc};
Point(9) = {0*hs,38,0,lc};
Point(10) = {0*hs,13,0,lc};
Point(11) = {15*hs,18,0,lc};
Point(12) = {25*hs,13,0,lc};
Point(13) = {45*hs,4,0,lc};
Point(14) = {54*hs,-7,0,lc};
Point(15) = {80*hs,-7,0,lc};
Point(16) = {97*hs,-18,0,lc};
Point(17) = {103*hs,-68,0,lc};
Point(18) = {130*hs,-68,0,lc};
Point(19) = {138*hs,-77,0,lc};

Line(1) = {1,2};
Line(2) = {2,3};
Spline(3) = {3,4,5};
Line(4) = {5,6};
Spline(5) = {6,7,8,9};
Line(6) = {9,10};
Spline(7) = {10,11,12,13,14};
Line(8) = {14,15};
Spline(9) = {15,16,17};
Spline(10) = {17,18,19};
Line(11) = {19,1};

Line Loop(12) = {2,3,4,5,6,7,8,9,10,11,1};
Plane Surface(13) = {12};

h = hs*50;

Extrude {0,0,h}{
    Surface{13};
    //Layers{5};
    //Recombine;
}
