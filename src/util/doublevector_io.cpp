#include <util/doublevector.h>

namespace util {

std::ostream& operator<<(std::ostream& os, const DoubleVector& v) {
    DoubleVector::size_type sz = v.size();
    os << "[ ";
    for (DoubleVector::size_type i = 0; i < sz; ++i) {
        os << v[i] << ' ';
    }
    os << "]";
    return os;
}

} // end namespace util
