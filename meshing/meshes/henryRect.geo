width = 200;
height = 100;
bottom = 0;
lc = height/21 - height/1000000.; // small adjustment to ensure correct number of mesh subdivisions

// Points in counterclockwise order
Point(1) = {0, bottom,      0, lc};
Point(2) = {0, bottom+height, 0, lc};

// lines
Line(1) = {2,1};

extSurf[] = Extrude {width,0,0} {Line{1}; Layers{width/lc}; Recombine;};

// the sides, which correspond to boundaries in the 2D model
Physical Line(1) = {extSurf[3], extSurf[2]};
Physical Line(2) = {extSurf[0]};
Physical Line(3) = {1};

// assign physical tag to the elements in the volume
Physical Surface(100) = {extSurf[1]};
