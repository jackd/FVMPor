//Geometry.Surfaces = 1;
//Geometry.SurfaceNumbers = 1;

// horizontal scale factor
hs = 1;

// meshing parameter
lc = 15;
lc1 = lc/2;
lc2 = lc/1.5;
lc3 = lc;

Point(1) = {160*hs,-77,0,lc1};
Point(2) = {160*hs,4,0,lc1};
Point(3) = {101*hs,4,0,lc1};
Point(4) = {98*hs,18,0,lc2};
Point(5) = {90*hs,25,0,lc2};
Point(6) = {60*hs,25,0,lc2};
Point(7) = {54*hs,28,0,lc2};
Point(8) = {38*hs,33,0,lc3};
Point(9) = {0*hs,38,0,lc3};
Point(10) = {0*hs,13,0,lc3};
Point(11) = {15*hs,18,0,lc3};
Point(12) = {25*hs,13,0,lc3};
Point(13) = {45*hs,4,0,lc2};
Point(14) = {54*hs,-7,0,lc2};
Point(15) = {80*hs,-7,0,lc2};
Point(16) = {97*hs,-18,0,lc1};
Point(17) = {103*hs,-68,0,lc1};
Point(18) = {130*hs,-68,0,lc1};
Point(19) = {138*hs,-77,0,lc1};
Point(20) = {45*hs,31,0,lc2};
Point(21) = {90*hs,-14,0,lc1};
Point(22) = {140*hs,-18,0,lc1};
Point(23) = {137*hs,-10,0,lc1};
Point(24) = {130*hs,-8,0,lc1};
Point(25) = {128*hs,-3,0,lc1};
Point(26) = {127*hs,-3,0,lc1};
Point(27) = {127*hs,4,0,lc1};
Point(28) = {133*hs,4,0,lc1};
Point(29) = {133*hs,-3,0,lc1};
Point(30) = {160*hs,-18,0,lc1};
xval = 97+3/25*14;
Point(31) = {xval*hs,-32,0,lc1};
Point(32) = {120*hs,-32,0,lc1};
Point(33) = {128*hs,-42,0,lc1};
Point(34) = {160*hs,-42,0,lc1};
Point(35) = {123*hs,-68,0,lc1};

// zone 1
Spline(1) = {20,8,9};
Line(2) = {9,10};
Spline(3) = {10,11,12,13};
Line(4) = {13,20};

// zone 2
Line(5) = {13,14};
Line(6) = {14,15};
Line(7) = {15,21};
Line(8) = {21,3};
Spline(9) = {3,4,5};
Line(10) = {5,6};
Spline(11) = {6,7,20};

// zone 3
Line(13) = {21,16};
Line(14) = {16,22};
Spline(15) = {22,23,24,25};
Line(16) = {25,26};
Line(17) = {26,27};
Line(18) = {27,3};

// zone 4
Line(19) = {25,29};
Line(20) = {29,28};
Line(21) = {28,27};

// zone 5
Line(22) = {22,30};
Line(23) = {30,2};
Line(24) = {2,28};

// zone 6
Line(25) = {16,31};
Line(26) = {31,32};
Line(27) = {32,33};
Line(28) = {33,34};
Line(29) = {34,30};

// zone 7
Line(30) = {31,17};
Line(31) = {17,35};
Line(32) = {35,32};

// zone 8
Line(33) = {35,18};
Line(34) = {18,19};
Line(35) = {19,1};
Line(36) = {1,34};

// define zone 1
Line Loop(101) = {1,2,3,4};
Plane Surface(201) = {101};

// define zone 2
Line Loop(102) = {5,6,7,8,9,10,11,-4};
Plane Surface(202) = {102};

// define zone 3
Line Loop(103) = {13,14,15,16,17,18,-8};
Plane Surface(203) = {103};

// define zone 4
Line Loop(104) = {-16,19,20,21,-17};
Plane Surface(204) = {104};

// define zone 5
Line Loop(105) = {22,23,24,-20,-19,-15};
Plane Surface(205) = {105};

// define zone 6
Line Loop(106) = {25,26,27,28,29,-22,-14};
Plane Surface(206) = {106};

// define zone 7
Line Loop(107) = {30,31,32,-26};
Plane Surface(207) = {107};

// define zone 8
Line Loop(108) = {-32,33,34,35,36,-28,-27};
Plane Surface(208) = {108};

// Extrude the surfaces

// set the "depth" of the domain
h = -hs*50;

// a list of the material properties of each volume
// these will have -100 applied in the Mesh.py
volProps[] = {100,101,102,103,104,105,106,107};

i = 1;
For k In {201:208}
    // extrude the volume
    out[] = Extrude {0,0,h}{
      Surface{k};
      //Layers{50/(lc)};
      //Recombine;
    };
    // tag the new volume with its material type
    Physical Volume(volProps[i-1]) = {out[1]};
    i++;

EndFor

// now set the physical faces for the boundary conditions
// i can't see how to automatically do this for extruded surfaces
// so we just do it manually
// BC type 1 - v dot n = given, dcdt=0
Physical Surface(1) = {351,396,455};
// BC type 2 - h = given, c=1
Physical Surface(2) = {217,267,263,259,304,331,355};
Physical Surface(3) = {221};
// BC type 4 - no flux
Physical Surface(4) = {201,202,203,204,205,206,207,208, 230,272,309,336,368,405,427,464, 225,243,247,251,284,380,414,418,443,447,451};
