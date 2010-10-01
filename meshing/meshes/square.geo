//Geometry.LineNumbers = 1;
Geometry.Surfaces = 1;
Geometry.SurfaceNumbers = 1;
Geometry.PointNumbers = 1;

// Gmsh project created on Tue Aug  5 16:03:10 2008
lc = 0.1;
Point(1) = {0,0,0,lc};
Point(2) = {1,0,0,lc};
Point(3) = {1,1,0,lc};
Point(4) = {0,1,0,lc};


Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,1};
Line(4) = {3,4};
Line(5) = {4,1};
Line Loop(6) = {1,2,3};
Line Loop(7) = {4,5,-3};
Plane Surface(8) = {6};
Plane Surface(9) = {7};
Recombine Surface{1,6} = 45;


nLayers = 10;

out[] = Extrude {0,0,1} {
  Surface{8};
  //Layers{nLayers};
  Recombine;
};
Physical Volume(102) = {out[1]};

out[] = Extrude {0,0,1} {
  Surface{9};
  //Layers{nLayers};
  Recombine;
};
Physical Volume(101) = {out[1]};
Physical Surface(1) = {8,9,26,119};
Physical Surface(2) = {114,17,21,110};

