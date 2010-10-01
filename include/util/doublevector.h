// type for storing and manipulating vectors of doubles
// ultimately this will want to have multiple implementations
// that call BLAS or have inline code for various operations.
#ifndef UTIL_DOUBLEVEC_H
#define UTIL_DOUBLEVEC_H

#include <vector>
#include <ostream>
#include <stdexcept>

#include <util/intvector.h>

namespace util {

class DoubleVector : private std::vector<double> {
    typedef std::vector<double> base;
public:
    using base::difference_type;

    // Constructors
    DoubleVector();
    explicit DoubleVector(difference_type count);
    DoubleVector(difference_type count, double val);
    template<typename Iterator>
    DoubleVector(Iterator, Iterator);

    // Iteration
    using base::begin;
    using base::end;
    using base::rbegin;
    using base::rend;

    // Size
    void reserve(difference_type);
    void resize(difference_type);
    void resize(difference_type, double);
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
    const double& operator[](difference_type) const;
    double& operator[](difference_type);

    // C++0x-style members
    const double* data() const;
    double* data();

    // Arithmetic
    DoubleVector& operator=(double);
    DoubleVector& operator+=(double);
    DoubleVector& operator-=(double);
    DoubleVector& operator*=(double);
    DoubleVector& operator/=(double);
    DoubleVector& raise_to(double);


    DoubleVector& operator+=(const DoubleVector&);
    DoubleVector& operator-=(const DoubleVector&);
    DoubleVector& operator*=(const DoubleVector&);
    DoubleVector& operator/=(const DoubleVector&);
    DoubleVector& raise_to(const DoubleVector&);
    DoubleVector& sqroot();
    DoubleVector& by_sqroot(const DoubleVector& v);
    DoubleVector& equals_prod(const DoubleVector&, const DoubleVector& );
    DoubleVector& plus_equals_prod(const DoubleVector&, const DoubleVector& );


    void permute_assign(const DoubleVector &, const IntVector &);
    void permute_assign_inverse(const DoubleVector &, const IntVector &);
    void permute_assign_inverse(const DoubleVector &, const DoubleVector &, const IntVector &);
    void permute_add_inverse(const DoubleVector &, const IntVector &);
    void permute_add_weighted_inverse(const DoubleVector &, const IntVector &, const DoubleVector &);
    void permute_add_weighted_pq(const DoubleVector &, const IntVector &, const IntVector &, const DoubleVector &);
    void permute_add_weighted_pqr(const DoubleVector &, const IntVector &, const IntVector &, const IntVector &,const DoubleVector &);
    void scal_assign(double, const DoubleVector &);
    void axpy(double, const DoubleVector &);

    // IO
    friend std::ostream& operator<<(std::ostream&, const DoubleVector&);
};

inline
DoubleVector::DoubleVector() {}

inline
DoubleVector::DoubleVector(difference_type count)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("DoubleVector::DoubleVector(count): count < 0");
    #endif
    base::resize(count);
}

inline
DoubleVector::DoubleVector(difference_type count, double val)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("DoubleVector::DoubleVector(count, val): count < 0");
    #endif
    base::resize(count, val);
}

template<typename Iterator>
DoubleVector::DoubleVector(Iterator first, Iterator last) : base(first, last) {}

inline
void DoubleVector::reserve(difference_type count)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("DoubleVector::reserve(count): count < 0");
    #endif
    base::reserve(count);
}

inline
void DoubleVector::resize(difference_type count)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("DoubleVector::resize(count): count < 0");
    #endif
    base::resize(count);
}

inline
void DoubleVector::resize(difference_type count, double val)
{
    #ifdef VECTOR_DEBUG
    if (count < 0) throw std::logic_error("DoubleVector::resize(count, val): count < 0");
    #endif
    base::resize(count, val);
}

inline
const double& DoubleVector::operator[](difference_type i) const
{
    #ifdef VECTOR_DEBUG
    if (i < 0) throw std::logic_error("DoubleVector::DoubleVector[i]: i < 0");
    if (i >= size()) throw std::logic_error("DoubleVector::DoubleVector[i]: i >= size()");
    #endif
    return base::operator[](i);
}

inline
double& DoubleVector::operator[](difference_type i)
{
    #ifdef VECTOR_DEBUG
    if (i < 0) throw std::logic_error("DoubleVector::DoubleVector[i]: i < 0");
    if (i >= size()) throw std::logic_error("DoubleVector::DoubleVector[i]: i >= size()");
    #endif
    return base::operator[](i);
}

inline
const double* DoubleVector::data() const {
    #ifdef VECTOR_DEBUG
    if (empty()) throw std::logic_error("DoubleVector::data(): empty()");
    #endif
    return &(*this)[0];
}

inline
double* DoubleVector::data() {
    #ifdef VECTOR_DEBUG
    if (empty()) throw std::logic_error("DoubleVector::data(): empty()");
    #endif
    return &(*this)[0];
}

} // end namespace util

#endif
