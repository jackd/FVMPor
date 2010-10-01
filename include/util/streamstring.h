#ifndef STREAMSTRING_H
#define STREAMSTRING_H
/****************************************************************
 * Helper functions and templates for strings and streams
 * Created 17-8-2010 Ben Cumming
 ***************************************************************/
#include <iostream>
#include <fstream>
#include <sstream>

namespace util{
    
// print to string
template<typename T> std::string to_string(const T& t){
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

// create a portable ostream equivalent to /dev/null
template <class cT, class traits = std::char_traits<cT> >
class basic_nullbuf: public std::basic_streambuf<cT, traits> {
    typename traits::int_type overflow(typename traits::int_type c)
    {
        return traits::not_eof(c); // indicate success
    }
};

template <class cT, class traits = std::char_traits<cT> >
class basic_onullstream: public std::basic_ostream<cT, traits> {
public:
    basic_onullstream():
    std::basic_ios<cT, traits>(&m_sbuf),
    std::basic_ostream<cT, traits>(&m_sbuf)
    {
        init(&m_sbuf);
    }

    private:
    basic_nullbuf<cT, traits> m_sbuf;
};

typedef basic_onullstream<char> onullstream;

} // namespace util
#endif
