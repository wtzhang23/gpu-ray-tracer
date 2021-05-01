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
class Material {
private:
    rmath::Vec4<float> Ke;
    rmath::Vec4<float> Ka;
    rmath::Vec4<float> Kd;
    rmath::Vec4<float> Ks;    
    rmath::Vec4<float> Kt;
    rmath::Vec4<float> Kr;
    float alpha;
    float eta;
public:
    CUDA_HOSTDEV
    Material(rmath::Vec4<float> Ke, rmath::Vec4<float> Ka, rmath::Vec4<float> Kd, rmath::Vec4<float> Ks, rmath::Vec4<float> Kt, rmath::Vec4<float> Kr,
                float alpha, float eta): Ke(Ke), Ka(Ka), Kd(Kd), Ks(Ks), Kt(Kt), Kr(Kr), alpha(alpha), eta(eta){}
    
    CUDA_HOSTDEV
    Material(): Ke(rmath::Vec4<float>({1.0f, 1.0f, 1.0f})), Ka(), Kd(), Ks(), Kt(), Kr(), alpha(0), eta(1){}

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Ke() const {
        return Ke;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Ka() const {
        return Ka;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Kd() const {
        return Kd;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Ks() const {
        return Ks;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Kt() const {
        return Kt;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Kr() const {
        return Kr;
    }

    CUDA_HOSTDEV
    float get_alpha() const {
        return alpha;
    }

    CUDA_HOSTDEV
    float get_eta() const {
        return eta;
    }
};
}

#endif