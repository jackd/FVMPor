#ifndef CHECKED_ITERATOR_H
#define CHECKED_ITERATOR_H

#include <stdexcept>
#include <iterator>

namespace util {

template<typename T>
class checked_iterator {

    template<typename U>
    struct remove_const {
        typedef U type;
    };
    
    template<typename U>
    struct remove_const<const U> {
        typedef U type;
    };

public:
    typedef typename remove_const<T>::type value_type;
    typedef std::ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T& reference;
    typedef std::random_access_iterator_tag iterator_category;

    checked_iterator() : it(), begin(), end() {}
    checked_iterator(T* it, T* begin, T* end)
        : it(it), begin(begin), end(end) {}
    template<typename U> checked_iterator(const checked_iterator<U>& other)
        :it(other.it), begin(other.begin), end(other.end) {}

    reference operator*() const {
        check_dereference();
        return *it;
    }

    pointer operator->() const {
        check_dereference();
        return it;
    }

    checked_iterator& operator++() {
        check_increment();
        ++it;
        return *this;
    }
    
    checked_iterator operator++(int) {
        check_increment();
        checked_iterator temp(*this);
        it++;
        return temp;
    }

    checked_iterator& operator--() {
        check_decrement();
        --it;
        return *this;
    }
    
    checked_iterator operator--(int) {
        check_decrement();
        checked_iterator temp(*this);
        it--;
        return temp;
    }

    checked_iterator operator+=(difference_type n) {
        check_advance(n);
        it += n;
    }

    checked_iterator operator-=(difference_type n) {
        check_advance(-n);
        it -= n;
    }

    reference operator[](difference_type n) const {
        check_advance(n);
        return *(it + n);
    }

    friend
    bool operator==(const checked_iterator& left, const checked_iterator& right)
    {
        check_related(left, right);
        return left.it == right.it;
    }

    friend
    bool operator!=(const checked_iterator& left, const checked_iterator& right)
    {
        check_related(left, right);
        return left.it != right.it;
    }

    friend
    bool operator<(const checked_iterator& left, const checked_iterator& right)
    {
        check_related(left, right);
        return left.it < right.it;
    }

    friend
    bool operator>(const checked_iterator& left, const checked_iterator& right)
    {
        check_related(left, right);
        return left.it > right.it;
    }

    friend
    bool operator<=(const checked_iterator& left, const checked_iterator& right)
    {
        check_related(left, right);
        return left.it <= right.it;
    }

    friend
    bool operator>=(const checked_iterator& left, const checked_iterator& right)
    {
        check_related(left, right);
        return left.it >= right.it;
    }

    friend
    checked_iterator operator+(const checked_iterator& left, difference_type n)
    {
        left.check_advance(n);
        return checked_iterator(left.it + n, left.begin, left.end);
    }

    friend
    checked_iterator operator+(difference_type n, const checked_iterator& right)
    {
        right.check_advance(n);
        return checked_iterator(n + right.it, right.begin, right.end);
    }

    friend
    checked_iterator operator-(const checked_iterator& left, difference_type n)
    {
        left.check_advance(-n);
        return checked_iterator(left.it - n, left.begin, left.end);
    }

    friend
    difference_type operator-(const checked_iterator& left,
                              const checked_iterator& right)
    {
        check_related(left, right);
        return left.it - right.it;
    }

private:
    template<typename U> friend class checked_iterator;

    void check_dereference() const {
        if (it == end)
            throw std::runtime_error("Attempt to dereference end");
    }

    void check_increment() const {
        if (it == end)
            throw std::runtime_error("Attempt to increment beyond end");
    }

    void check_decrement() const {
        if (it == begin)
            throw std::runtime_error("Attempt to decrement beyond begin");
    }

    void check_advance(difference_type n) const {
        if (n > 0 && end - it < n)
            throw std::runtime_error("Attempt to advance beyond end");
        if (n < 0 && it - begin < -n)
            throw std::runtime_error("Attempt to retreat beyond begin");
    }

    static void check_related(const checked_iterator& left,
                              const checked_iterator& right)
    {
        if (left.begin != right.begin || left.end != right.end)
            throw std::runtime_error("Attempt to mix unrelated iterators");
    }

    T* it;
    T* begin;
    T* end;

};

} // end namespace util

#endif
