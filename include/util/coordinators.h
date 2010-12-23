#ifndef COORDINATORS_H
#define COORDINATORS_H

#include <lin/coordinators/gpu/coordinator.h>
#include <lin/lin.h>

namespace util{

template <typename T>
struct CoordTraits{
    static bool is_device() {return false;};
};
template <>
struct CoordTraits<lin::gpu::Coordinator<int> >{
    static bool is_device() {return true;};
};
template <>
struct CoordTraits<lin::gpu::Coordinator<double> >{
    static bool is_device() {return true;};
};

} // namespace util

#endif
