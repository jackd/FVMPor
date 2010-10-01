#include "fvmpor.h"
#include "shape.h"

#include <mkl_vml_functions.h>

#include <util/doublevector.h>
#include <fvm/mesh.h>

#include <algorithm>
#include <vector>
#include <limits>
#include <utility>

#include <math.h>

// For debugging
#include <iostream>
#include <ostream>
#include <iterator>

namespace fvmpor {


    double porosity(double h, double por_0, const Constants& constants)
    {
        return por_0;
    }

} // end namespace fvmpor
