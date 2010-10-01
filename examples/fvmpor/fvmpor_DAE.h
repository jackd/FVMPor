#ifndef FVMPOR_DAE_H
#define FVMPOR_DAE_H

#include "fvmpor.h"
#include <util/doublevector.h>
#include <iomanip>

namespace fvmpor {

struct hM {
    double h;
    double M;
    // helpers
    static const int variables = 2;
    static const int differential_variables = 1;
    static std::string var_name(int i){
        switch(i){
            case 0:
                return std::string("pressure_head");
            case 1:
                return std::string("fluid_mass");
        }
        return std::string("unknownVariable");
    }
    friend std::ostream& operator<<(std::ostream& os,
                                    const hM& val)
    {
        return os  << val.h << "    " << val.M;
    }
};

typedef VarSatPhysics<hM, util::DoubleVector> Physics;

} // end namespace fvmpor

#endif

