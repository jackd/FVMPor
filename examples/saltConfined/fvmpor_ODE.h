#ifndef FVMPOR_ODE_H
#define FVMPOR_ODE_H

#include "fvmpor.h"
#include <util/doublevector.h>

namespace fvmpor {

// holds two variables : pressure head and salt concentration
struct hc{
    static const int variables = 2;
    static const int differential_variables = 2;
    double h;
    double c;
    friend std::ostream& operator<<(std::ostream& os,
                                    const Head& val)
    {
        return os << val.h;
    }
};

typedef VarSatPhysics<hc, util::DoubleVector> Physics;

} // end namespace fvmpor

#endif

