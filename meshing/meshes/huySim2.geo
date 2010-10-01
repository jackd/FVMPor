//lc = 12.5 - 1e-8; // small adjustment to ensure correct number of mesh subdivisions
lc = 2 - 1e-8; // small adjustment to ensure correct number of mesh subdivisions
//width = 200;
width = 50;
height = 50;
topHeight = height;

// Points in counterclockwise order
Point(1) = {0, 0,      0, lc};
Point(2) = {0, height, 0, lc};

// lines
Line(1) = {2,1};

extSurf[] = Extrude {width,0,0} {Line{1}; Layers{width/lc}; Recombine;};

// the sides, which correspond to boundaries in the 2D model
Physical Line(1) = {1};
Physical Line(2) = {extSurf[3]};
Physical Line(3) = {extSurf[0]};
Physical Line(4) = {extSurf[2]};

// assign physical tag to the elements in the volume
Physical Surface(100) = {extSurf[1]};
