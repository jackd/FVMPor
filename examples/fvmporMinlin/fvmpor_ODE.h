#ifndef FVMPOR_ODE_H
#define FVMPOR_ODE_H

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

typedef lin::DefaultCoordinator<int> CPUCoord;
typedef lin::gpu::Coordinator<int> GPUCoord;

typedef VarSatPhysics<Head, CPUCoord, GPUCoord> PhysicsGPU;
typedef VarSatPhysics<Head, CPUCoord, CPUCoord> PhysicsCPU;
typedef PhysicsCPU Physics;

} // end namespace fvmpor

#endif

