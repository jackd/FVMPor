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
    typedef typename fvm::Callback<Physics> Callback;
    typedef typename fvm::Iterator<value_type>::type iterator;
    typedef typename fvm::ConstIterator<value_type>::type const_iterator;
};

} // end namespace fvm

#endif
