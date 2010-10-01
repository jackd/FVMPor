#ifndef FVMPOR_ODE_H
#define FVMPOR_ODE_H

#include "fvmpor.h"
#include <util/doublevector.h>

namespace fvmpor {

struct Head{
    double h;
    // helpers
    static const int variables = 1;
    static const int differential_variables = 1;
    static std::string var_name(int i){
        switch(i){
            case 0:
                return std::string("pressure_head");
        }
        return std::string("unknownVariable");
    }
    friend std::ostream& operator<<(std::ostream& os,
                                    const Head& val)
    {
        return os << val.h;
    }
};

typedef VarSatPhysics<Head, util::DoubleVector> Physics;

} // end namespace fvmpor

#endif

