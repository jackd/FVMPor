#ifndef MESH_EXCEPTION_H
#define MESH_EXCEPTION_H

#include <stdexcept>
#include <string>

namespace mesh {

class Exception {
public:
    virtual const char* what() const throw() = 0;
    virtual ~Exception() throw() {}
};

class OutOfRangeException : public Exception, public std::logic_error {
public:
    OutOfRangeException(const std::string& message)
        : std::logic_error("OutOfRange: " + message) {}
    virtual const char* what() const throw() {
        return std::logic_error::what();
    }
    virtual ~OutOfRangeException() throw() {}
};

class IOException : public Exception, public std::runtime_error {
public:
    IOException(const std::string& message)
        : std::runtime_error("IO: " + message) {}
    virtual const char* what() const throw() {
        return std::runtime_error::what();
    }
    virtual ~IOException() throw() {}
};

} // end namespace mesh

#endif
