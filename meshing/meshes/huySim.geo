lc = 5 - 1e-8; // small adjustment to ensure correct number of mesh subdivisions
width = 200;
height = 50;
topHeight = 50;
nLayers3D = 1;

// Points in counterclockwise order
Point(1) = {0, 0, 0, lc};
Point(2) = {0, 0, height, lc};

// lines
Line(1) = {2,1};

//extSurf[] = Extrude {width,0,0} {Line{1}; Layers{width/lc}; Recombine;};
//extVol[] = Extrude {0,lc*nLayers3D,0} {Surface{extSurf[1]}; Layers{nLayers3D}; Recombine;};
extSurf[] = Extrude {width,0,0} {Line{1}; Layers{width/lc}; };
extVol[] = Extrude {0,lc*nLayers3D,0} {Surface{extSurf[1]}; Layers{nLayers3D};};


// the front and back of the domain
Physical Surface(6) = {extSurf[1],extVol[0]};
// the sides, which correspond to boundaries in the 2D model
Physical Surface(1) = {extVol[2]};
Physical Surface(2) = {extVol[5]};
Physical Surface(3) = {extVol[4]};
Physical Surface(4) = {extVol[3]};

// assign physical tag to the elements in the volume
Physical Volume(101) = {extVol[1]};
