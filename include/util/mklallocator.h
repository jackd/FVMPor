#ifndef MKLALLOCATOR_H
#define MKLALLOCATOR_H

#include <mkl_service.h>
#include <memory>

namespace util {

template<typename T>
class MKLAllocator : public std::allocator<T> {
public:
    MKLAllocator() throw() {}
    template<typename U>
    MKLAllocator(const MKLAllocator<U>& other) throw() {}
    template<typename U>
    struct rebind {
        typedef MKLAllocator<U> other;
    };

    T* allocate(std::size_t count) {
        T* p = static_cast<T*>(MKL_malloc(count*sizeof(T), 16));
        assert(p);
        return p;
    }

    void deallocate(T* p, std::size_t) {
        MKL_free(p);
    }

};

} // namespace util

#endif
