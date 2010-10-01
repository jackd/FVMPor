#ifndef FVMPOR_DAE_H
#define FVMPOR_DAE_H

#include "fvmpor.h"
#include <util/doublevector.h>
#include <iomanip>

namespace fvmpor {

struct hcMC {
    static const int variables = 4;
    static const int differential_variables = 2;
    double h;
    double c;
    double M;
    double C;
    friend std::ostream& operator<<(std::ostream& os,
                                    const hcMC& val)
    {
        return os  << val.h << " " << val.c << val.M << " " << val.C;
    }
    std::string var_name(int i){
        switch(i){
            case 0:
                return std::string("pressure_head");
            case 1:
                return std::string("salt_concentration");
            case 2:
                return std::string("fluid_mass");
            case 3:
                return std::string("salt_mass");
        }
        return std::string("unknownVariable");
    }
};

typedef DensityDrivenPhysics<hcMC, util::DoubleVector> Physics;

} // end namespace fvmpor

#endif

