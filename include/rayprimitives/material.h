#ifndef MATERIAL_H
#define MATERIAL_H

#include "raymath/linear.h"
#include "rayprimitives/texture.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
template <typename T>
class Material {
private:
    rmath::Vec3<T> Ke;
    rmath::Vec3<T> Kd;
    rmath::Vec3<T> Ks;    
    rmath::Vec3<T> Kt;
    rmath::Vec3<T> Kr;
    T alpha;
    T eta;
public:
    Material(rmath::Vec3<T> Ke, rmath::Vec3<T> Kd, rmath::Vec3<T> Ks, rmath::Vec3<T> Kt, rmath::Vec3<T> Kr,
                T alpha, T eta): Ke(Ke), Kd(Kd), Ks(Ks), Kt(Kt), Kr(Kr), alpha(alpha), eta(eta){}
    Material(): Material(rmath::Vec3<T>(1, 1, 1), rmath::Vec3<T>(0, 0, 0), rmath::Vec3<T>(0, 0, 0),
                rmath::Vec3<T>(0, 0, 0), rmath::Vec3<T>(0, 0, 0), rmath::Vec3<T>(0, 0, 0)){}

};
}

#endif