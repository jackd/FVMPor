#ifndef PRECONDITIONER_BASE_H
#define PRECONDITIONER_BASE_H

#include "fvm.h"

namespace fvm {

// A base class template for Preconditioner classes.  It provides a few simple
// typedefs.

template<typename Physics>
class PreconditionerBase {
public:
    typedef typename Physics::value_type value_type;
    typedef typename Physics::TVec TVec;
    typedef typename Physics::TVecDevice TVecDevice;
    typedef typename fvm::Callback<Physics> Callback;
};

} // end namespace fvm

#endif
