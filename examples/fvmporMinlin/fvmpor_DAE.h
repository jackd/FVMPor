#ifndef FVMPOR_DAE_H
#define FVMPOR_DAE_H

#include "fvmpor.h"
#include <lin/lin.h>
#include <lin/coordinators/gpu/coordinator.h>

namespace fvmpor {

template <typename T>
struct CoordTraits{
    static bool is_device() {return false;};
};
template <>
struct CoordTraits<lin::gpu::Coordinator<int> >{
    static bool is_device() {return true;};
};

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

typedef lin::DefaultCoordinator<int> CPUCoord;
typedef lin::gpu::Coordinator<int> GPUCoord;

typedef VarSatPhysics<hM, CPUCoord, GPUCoord> PhysicsGPU;
typedef VarSatPhysics<hM, CPUCoord, CPUCoord> PhysicsCPU;
typedef PhysicsGPU Physics;

} // end namespace fvmpor

#endif

