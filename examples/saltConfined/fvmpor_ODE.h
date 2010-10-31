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

// holds two variables : pressure head and salt concentration
struct hc{
    static const int variables = 2;
    static const int differential_variables = 1;
    double c;
    double h;
    friend std::ostream& operator<<(std::ostream& os,
                                    const hc& val)
    {
        return os << val.h;
    }
    static std::string var_name(int i){
        switch(i){
            case 0:
                return std::string("concentration");
            case 1:
                return std::string("pressure_head");
        }
        return std::string("unknownVariable");
    }
};

typedef lin::DefaultCoordinator<int> CPUCoord;
typedef lin::gpu::Coordinator<int> GPUCoord;

typedef DensityDrivenPhysics<hc, CPUCoord, GPUCoord> PhysicsGPU;
typedef DensityDrivenPhysics<hc, CPUCoord, CPUCoord> PhysicsCPU;
//typedef PhysicsGPU Physics;
typedef PhysicsCPU Physics;

} // end namespace fvmpor

#endif

