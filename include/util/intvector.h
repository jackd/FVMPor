// type for storing and manipulating vectors of ints
// ultimately this will want to have multiple implementations
// that call BLAS or have inline code for various operations.
#ifndef UTIL_INTVECTOR_H
#define UTIL_INTVECTOR_H

#include <vector>
#include <ostream>
#include <stdexcept>

namespace util {

class IntVector : private std::vector<int> {
    typedef std::vector<int> base;
public:
    using base::difference_type;

    // Constructors
    IntVector();
    explicit IntVector(difference_type count);
    IntVector(difference_type count, int val);
    template<typename Iterator>
    IntVector(Iterator, Iterator);

    // Iteration
    using base::begin;
    using base::end;
    using base::rbegin;
    using base::rend;

    // Size
    void reserve(difference_type);
    void resize(difference_type);
    void resize(difference_type, int);
    using base::empty;

    // Front and back
    using base::front;
    using base::back;
    using base::push_back;
    using base::pop_back;

    // size of vector
    using base::size;

    // Clear
    using base::clear;

    // Swap
    using base::swap;

    // Indexing
    const int& operator[](difference_type) const;
    int& operator[](difference_type);

    // C++0x-style members
    const int* data() const;
    int* data();

    // Arithmetic
    //IntVector& operator=(int);
    //IntVector& operator=(const IntVector&);

    // IO
    friend std::ostream& operator<<(std::ostream&, const IntVector&);
};

inline
IntVector::IntVector() {}

inline
IntVector::IntVector(difference_type count)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("IntVector::IntVector(count): count < 0");
    #endif
    base::resize(count);
}

inline
IntVector::IntVector(difference_type count, int val)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("IntVector::IntVector(count, val): count < 0");
    #endif
    base::resize(count, val);
}

template<typename Iterator>
IntVector::IntVector(Iterator first, Iterator last) : base(first, last) {}

inline
void IntVector::reserve(difference_type count)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("IntVector::reserve(count): count < 0");
    #endif
    base::reserve(count);
}

inline
void IntVector::resize(difference_type count)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("IntVector::resize(count): count < 0");
    #endif
    base::resize(count);
}

inline
void IntVector::resize(difference_type count, int val)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("IntVector::resize(count, val): count < 0");
    #endif
    base::resize(count, val);
}

inline
const int& IntVector::operator[](difference_type i) const
{
    #ifdef VECTOR_DEBUG
    if (i < 0) throw std::logic_error("IntVector::IntVector[i]: i < 0");
    if (i >= size()) throw std::logic_error("IntVector::IntVector[i]: i >= size()");
    #endif
    return base::operator[](i);
}

inline
int& IntVector::operator[](difference_type i)
{
    #ifdef VECTOR_DEBUG
    if (i < 0) throw std::logic_error("IntVector::IntVector[i]: i < 0");
    if (i >= size()) throw std::logic_error("IntVector::IntVector[i]: i >= size()");
    #endif
    return base::operator[](i);
}

inline
const int* IntVector::data() const {
    #ifdef VECTOR_DEBUG
    if (empty()) throw std::logic_error("IntVector::data(): empty()");
    #endif
    return &(*this)[0];
}

inline
int* IntVector::data() {
    #ifdef VECTOR_DEBUG
    if (empty()) throw std::logic_error("IntVector::data(): empty()");
    #endif
    return &(*this)[0];
}

} // end namespace util

#endif
