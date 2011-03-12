#ifndef SHAPE_H
#define SHAPE_H

#include <fvm/mesh.h>
#include <util/point.h>
#include <vector>

#include <cassert>

namespace shape {

class Shape {
public:
    typedef std::vector<double> shape_function_vector;
    typedef std::vector<mesh::Point>  shape_gradient_vector;

    Shape(const mesh::Element&);

    shape_function_vector shape_functions(int edge) const {
        assert(edge >= 0 && edge < shape_f.size());
        return shape_f[edge];
    }
    shape_gradient_vector shape_gradients(int edge) const {
        assert(edge >= 0 && edge < shape_g.size());
        return shape_g[edge];
    }

private:
    static const int tet_nodes = 4;
    static const int brick_nodes = 8;
    static const int tet_edges = 6;
    static const int brick_edges = 12;
    std::vector<shape_function_vector> shape_f;
    std::vector<shape_gradient_vector> shape_g;
    static double det3(double, double, double,
                      double, double, double,
                      double, double, double);
    static double det2(double, double,
                      double, double);
};

} // end namespace shape

#endif
