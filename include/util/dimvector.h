#ifndef MINLIN_H
#define MINLIN_H
namespace util{

// used to store vector quantities such as fluxes that are
// defined over multiple nodes/faces/points in the mesh
template <typename TVec>
class DimVector{

    public:
        DimVector(int n, int dim);
        DimVector(): dim_(0){};
        void set(int n, int dim);
        TVec& x() {assert(dim_); return x_;};
        TVec& y() {assert(dim_); return y_;};
        TVec& z() {assert(dim_); return z_;};
        int dim() {assert(dim_); return dim;};

        void dot(DimVector<TVec> &normals, TVec &result);

    private:
        TVec x_;
        TVec y_;
        TVec z_;
        int dim_;
};

template <typename TVec>
DimVector<TVec>::DimVector(int n, int dim){
            set(n, dim);
}

template <typename TVec>
void DimVector<TVec>::set(int n, int dim){
    dim_ = dim;
    assert(n>0);
    assert(dim_==2 || dim_==3);
#ifdef USE_MINLIN
    x_ = TVec(n,0.);
    y_ = TVec(n,0.);
    if( dim_==3 )
        z_ = TVec(n,0.);
#else
    x_.resize(n);
    y_.resize(n);
    if( dim_==3 )
        z_.resize(n);
#endif
}

template <typename TVec>
void DimVector<TVec>::dot(DimVector<TVec> &normals, TVec &result){
    assert(dim_);
#ifdef USE_MINLIN
    result = mul(x_, normals.x());
    result += mul(y_, normals.y());
    if( dim_==3 )
        result += mul(z_, normals.z());
#else
    result.equals_prod(x_, normals.x());
    result.plus_equals_prod(y_, normals.y());
    if( dim_==3 )
        result.plus_equals_prod(z_, normals.z());
#endif
}

}

#endif
